import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from net.unit.base import ConvBNReLU3D


class RegularNet_3Scales(nn.Module):
    def __init__(self,
                 in_chs: int=8,
                 inner_chs: int=16,
                 ) -> None:
        super(RegularNet_3Scales, self).__init__()
        c0, c1, c2, = inner_chs, inner_chs*2, inner_chs*4

        self.conv01 = nn.Sequential(
            ConvBNReLU3D(in_chs, c0, kernel_size=3, padding=1),
            ConvBNReLU3D(c0, c0, kernel_size=3, padding=1),
        )

        self.conv12 = nn.Sequential(
            ConvBNReLU3D(c0, c1, kernel_size=3, stride=2, padding=1),
            ConvBNReLU3D(c1, c1, kernel_size=3, padding=1),
            ConvBNReLU3D(c1, c1, kernel_size=3, padding=1),
        )

        self.conv232 = nn.Sequential(
            ConvBNReLU3D(c1, c2, kernel_size=3, stride=2, padding=1),
            ConvBNReLU3D(c2, c2, kernel_size=3, padding=1),
            ConvBNReLU3D(c2, c2, kernel_size=3, padding=1),
            nn.ConvTranspose3d(c2, c1, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose3d(c1, c0, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(c0),
            nn.ReLU(inplace=True),
        )

        self.prob = nn.Conv3d(c0, 1, 3, stride=1, padding=1, bias=False)

        print('{} parameters: {}'.format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        """
        3D CNN regular
        @param x: cost volume:(B,C,D,H,W)
        @return: prob volume；(B,D,H,W)
        """
        B, C, D, H, W = x.shape
        assert (H%4==0 and W%4==0), "the shape of input volume must can div 4!"+"current cost volume shape:"+str(x.shape)

        # index = [i for i, c in enumerate(self.possible_inchs) if C==c ][0]
        # x = self.conv01_1(self.conv01_0[index](x))
        x = self.conv01(x)
        x1 = self.conv12(x)

        x1 = x1+self.conv232(x1)
        x = x+self.conv10(x1)
        del x1

        x = self.prob(x).squeeze(1)

        return F.softmax(x, dim=1)


class RegularNet_4Scales(nn.Module):
    def __init__(self,
                 in_chs: int,
                 base_chs: int = 8,
                 sample_stride: Tuple = (2, 2, 2),  #(1, 2, 2)
                 sample_padding: Tuple = (1, 1, 1), #(0, 1, 1)
                 ):
        super(RegularNet_4Scales, self).__init__()
        c0, c1, c2, c3 = base_chs, base_chs*2, base_chs*4, base_chs*8

        self.conv01 = ConvBNReLU3D(in_chs, c0, kernel_size=3, padding=1)

        self.conv12 = nn.Sequential(
            ConvBNReLU3D(c0, c1, 3, sample_stride, 1),
            ConvBNReLU3D(c1, c1, 3, 1, 1),
        )
        self.conv23 = nn.Sequential(
            ConvBNReLU3D(c1, c2, 3, sample_stride, 1),
            ConvBNReLU3D(c2, c2, 3, 1, 1),
        )
        self.conv343 = nn.Sequential(
            ConvBNReLU3D(c2, c3, 3, sample_stride, 1),
            ConvBNReLU3D(c3, c3, 3, 1, 1),
            nn.ConvTranspose3d(c3, c2, kernel_size=3, padding=1, output_padding=sample_padding, stride=sample_stride, bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
        )
        self.trconv32 = nn.Sequential(
            nn.ConvTranspose3d(c2, c1, kernel_size=3, padding=1, output_padding=sample_padding, stride=sample_stride, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
        )
        self.trconv21 = nn.Sequential(
            nn.ConvTranspose3d(c1, c0, kernel_size=3, padding=1, output_padding=sample_padding, stride=sample_stride, bias=False),
            nn.BatchNorm3d(c0),
            nn.ReLU(inplace=True),
        )

        self.prob = nn.Conv3d(c0, 1, 3, stride=1, padding=1, bias=False)

        print('{} parameters: {}'.format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self,
                x: torch.Tensor
                ):
        B, C, D, H, W = x.shape
        assert (H % 8 == 0 and W % 8 == 0), \
            "the shape of input volume must can div 8!" + "current cost volume shape:" + str(x.shape)

        x1 = self.conv01(x)
        x2 = self.conv12(x1)
        x3 = self.conv23(x2)

        x3 = x3 + self.conv343(x3)
        x2 = x2 + self.trconv32(x3)
        del x3
        x1 = x1 + self.trconv21(x2)
        del x2
        x = self.prob(x1).squeeze(1)
        del x1

        return F.softmax(x, dim=1)


if __name__=="__main__":
    import time
    s = time.time()
    r =RegularNet_4Scales() #possible_inchs=[8,], inner_chs=8
    x = torch.randn(4,16,8,160,128)
    y = r(x)
    print(y.shape, time.time()-s)
