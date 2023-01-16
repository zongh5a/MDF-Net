import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBNReLU(nn.Module):
    def __init__(self,
                 inchs: int,
                 outchs: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 ) -> None:
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(inchs, outchs, kernel_size, stride, (kernel_size-1)//2, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(outchs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class TrConvBNReLU(nn.Module):
    def __init__(self,
                 inchs: int,
                 outchs: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 output_padding: int=1,
                 groups: int = 1,
                 bias: bool = False,
                 ) -> None:
        super(TrConvBNReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(inchs, outchs, kernel_size, stride, padding, output_padding, groups, bias)
        self.bn = nn.BatchNorm2d(outchs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvBNReLU3D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=3,
                 stride=1,
                 padding: int=1,
                 groups: int=1,
                 bias: bool=False,
                 ) -> None:
        super(ConvBNReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Res(nn.Module):
    def __init__(self, chs, ):
        super(Res, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chs, chs, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(chs, chs, 3, 1, 1, bias=False),
        )

    def forward(self,x):

        return x + self.conv(x) * 0.1


def homo_warping(src_fea, src_proj, ref_proj, depth_hypos):
    """

    @param src_fea: [B, C, H, W]
    @param src_proj: [B, 4, 4]
    @param ref_proj: [B, 4, 4]
    @param depth_hypos: [B, Ndepth, 1, 1] or [B,NDepths,H,W]
    @return: [B, C, Ndepth, H, W]
    """
    batch, ndepths, H, W = depth_hypos.shape
    batch, channels ,height, width= src_fea.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        # del x, y

        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, ndepths, 1) * depth_hypos.view(batch, 1, ndepths, H * W) # [B, 3, Ndepth, H*W]

        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        # del rot_xyz, rot_depth_xyz, proj_xyz, proj_x_normalized, proj_y_normalized

    warped_src_fea = \
        F.grid_sample(src_fea, proj_xy.view(batch, ndepths * height, width, 2), mode='bilinear',padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, ndepths, height, width)

    return warped_src_fea


if __name__=="__main__":
    pass
