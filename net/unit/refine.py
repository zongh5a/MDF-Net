import torch
import torch.nn as nn
import torch.nn.functional as F

from net.unit.base import ConvBNReLU, Res


class RefineNet2(nn.Module):
    def __init__(self,
                 base_chs: int=8,
                 nres: int=3,
                 ):
        super(RefineNet2, self).__init__()
        self.conv0 = nn.Conv2d(1, base_chs, 3, 1, 1, bias=False)
        self.ress = nn.ModuleList([Res(base_chs) for _ in range(nres)])
        self.conv1 = nn.Conv2d(base_chs, base_chs, 3, 1, 1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_chs, base_chs * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(base_chs, 1, 3, 1, 1, bias=False)
        )

        print('{} parameters: {}'.format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self,
                depth: torch.Tensor,
                depth_range: torch.Tensor,
                ) -> torch.Tensor:
        depth = depth.unsqueeze(1).detach()
        B, _, _, _ = depth.shape
        depth_min, depth_max = depth_range[:, 0].float(), depth_range[:, 1].float()
        # pre-scale the depth map into [0,1]
        depth = (depth - depth_min.view(B, 1, 1, 1)) / ((depth_max - depth_min).view(B, 1, 1, 1)) #* 10

        # upscale
        depth = self.conv0(depth)
        d = depth
        for res in self.ress:
            depth = res(depth)
        depth = self.conv1(depth)
        depth = self.conv2(d + depth)

        depth =  depth_min.view(B, 1, 1, 1)+\
                 depth * (depth_max.view(B, 1, 1, 1) - depth_min.view(B, 1, 1, 1))

        return depth.squeeze(1)


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv_img = ConvBNReLU(3, 8)
        self.conv_depth = nn.Sequential(
            ConvBNReLU(1, 8),
            ConvBNReLU(8, 8),
            nn.ConvTranspose2d(8, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.conv_res = nn.Sequential(
            ConvBNReLU(16, 8),
            nn.Conv2d(8, 1, 3, 1, 1, bias=False),
        )

        print('{} parameters: {}'.format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self,
                ref_img: torch.Tensor,
                depth: torch.Tensor,
                depth_range: torch.Tensor,
                ) -> torch.Tensor:
        """

        @param ref_img: (B, 3, H, W)
        @param depth: (B, 1, H/2, W/2)
        @param depth_range: (B, 2)   B*(depth_min, depth_max)
        @return:depth map (B, H, W)
        """
        B, _, H, W = ref_img.shape
        depth = depth.unsqueeze(1).detach()
        depth_min, depth_max = depth_range[:, 0].float(), depth_range[:, 1].float()
        # pre-scale the depth map into [0,1]
        depth = (depth - depth_min.view(B, 1, 1, 1)) / ((depth_max - depth_min).view(B, 1, 1, 1)) #* 10

        ref_img = self.conv_img(ref_img)
        depth_conv = self.conv_depth(depth)

        res = self.conv_res(torch.cat([ref_img,depth_conv], dim=1))
        depth = F.interpolate(depth, scale_factor=2, mode="bilinear", align_corners=True) + res
        # convert the normalized depth back
        depth =  depth_min.view(B, 1, 1, 1)+\
                 depth * (depth_max.view(B, 1, 1, 1) - depth_min.view(B, 1, 1, 1))

        return depth.squeeze(1)


if __name__=="__main__":
    r =RefineNet2()
    img = torch.randn(2, 3, 640, 512)
    depth = torch.randn(2, 1, 320, 256)
    depth_range = torch.randn(2, 2)
    # depth = r(img, depth, depth_range)
    depth = r( depth, depth_range)

    print(depth.shape)