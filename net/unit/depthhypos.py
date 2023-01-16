import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


"""
4/29
"""
class HyposByFit(nn.Module):
    def __init__(self,
                 ndepths: int = 16,
                 curve_calss: str = "gauss1",
                 prob_thresh: float = 0.95,
                 ) -> None:
        super(HyposByFit, self).__init__()
        """

        @param ndepths:
        @param curve_calss: "gauss0"|"gauss1"|"laplace"
        @param prob_thresh:
        """
        self.ndepths, self.curve_calss, self.prob_thresh = \
            ndepths, curve_calss, torch.tensor(prob_thresh)


    def forward(self, depth, depth_range, prob_volume, depth_hypos, upsample = False):
        B = depth_range.shape[0]
        depth_min, depth_max = depth_range[:, 0].float(), depth_range[:, 1].float()
        # init hypos
        if depth is None:
            depth_min, depth_max = depth_min.view(B, 1, 1, 1), depth_max.view(B, 1, 1, 1)
            depth_interval = (depth_max - depth_min) / (self.ndepths - 1)
            depth_hypos = depth_min.unsqueeze(1) \
                          + (torch.arange(0, self.ndepths, device=depth_range.device).reshape(1, -1)
                             * depth_interval.unsqueeze(1))

            return depth_hypos.view(B, self.ndepths, 1, 1)

        with torch.no_grad():
            # fit curve
            if self.curve_calss == "gauss0":
                s = self._gauss_fitting0(depth, prob_volume, depth_hypos)
            elif self.curve_calss == "gauss1":
                s = self._gauss_fitting1(depth, prob_volume, depth_hypos)
            elif  self.curve_calss == "laplace":
                s = self._laplace_fitting(depth, prob_volume, depth_hypos)

            if upsample:
                s = F.interpolate(s.unsqueeze(1), scale_factor=2, mode='bilinear').squeeze(1)
                depth = F.interpolate(depth.unsqueeze(1), scale_factor=2, mode='bilinear').squeeze(1)

            # generate depth res
            if self.curve_calss == "gauss0" or self.curve_calss == "gauss1":
                depth_res = torch.sqrt(-1*s*torch.log(self.prob_thresh))
            elif self.curve_calss == "laplace":
                depth_res = torch.abs(s * torch.log(self.prob_thresh))
            depth_res = depth_res.clamp(min=1e-6, max=(depth_max.max()-depth_min.min())/2)

            for b in range(B):
                depth_res[b, ::] = depth_res[b, ::].clamp(max= (depth_max[b]-depth_min[b]).mean()*0.2)

            # generate depth hypos
            intervals = (depth_res / (self.ndepths - 1))
            depth_hypos = (depth - 0.5 * depth_res).unsqueeze(1).repeat(1, self.ndepths, 1, 1)  # res_min<0
            for d in range(self.ndepths):
                depth_hypos[:, d, :, :] += intervals * d

            # clamp
            for b in range(B):
                delta = (depth_hypos[b] - depth_min[b]).clamp(min=0)
                depth_hypos[b] = depth_min[b] + delta
                delta = (depth_hypos[b] - depth_max[b]).clamp(max=0)
                depth_hypos[b] = depth_max[b] + delta

            return depth_hypos  # .to(depth_range.device)

    def _laplace_fitting(
            self,
            depth: torch.Tensor,
            prob_volume: torch.Tensor,
            depth_hypos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fitting probability curve: x is depth hypos, y is preb value,
        Z = lny
        X = [[abs(x1-depth)]
             [  ...    ]
             [abs(xn-depth)]]
        B = (X.T.X)-1.X.T.Z     [b0,]
        b = -1/b0
        use: res = torch.abs(-1*s*torch.log(prob_values[n]))
        @param prob_volume:(B, D, H, W)
        @param depth_hypos: (B, D, H, W)|(B, D, 1, 1)
        """
        B, D, H, W = prob_volume.shape
        if depth_hypos.shape[-1] != prob_volume.shape[-1]:
            depth_hypos = depth_hypos.view(B, D, 1, 1).repeat(1, 1, H, W)

        # Z
        # prob_volume = prob_volume.clamp(min=1e-6)  # very key 40
        # Z = torch.log(prob_volume).unsqueeze(-1).permute(0, 2, 3, 1, 4)  # (B, D, H, W, 1) -> (B, H, W, D, 1)
        #
        # # X
        # X = torch.abs(depth_hypos - depth.unsqueeze(1)).unsqueeze(-1).permute(0, 2, 3, 1, 4)  # (B, D, H, W, 1) -> (B, H, W, D, 1)
        # X_T = X.transpose(-1, -2)
        #
        # # B
        # X = torch.matmul(torch.inverse(torch.matmul(X_T, X)), X_T)
        # B = torch.matmul(X, Z).squeeze(-1)  # (B, H, W, 3)
        #
        # # s , u: likely <0 -> abs
        # b = torch.abs(-1 / B[:, :, :, 0])  # (B, H, W)

        # more fast, not use matrix
        prob_volume = prob_volume.clamp(min=1e-40)
        y = torch.log(prob_volume).permute(0, 2, 3, 1)
        x = torch.abs(depth_hypos - depth.unsqueeze(1)).permute(0, 2, 3, 1)
        sum_xy = torch.sum(x*y, dim=-1)
        sum_xx = torch.sum(x*x, dim=-1)

        b = torch.abs(sum_xy/sum_xx)
        b = 1/b

        return b

    def _gauss_fitting0(
            self,
            depth: torch.Tensor,
            prob_volume: torch.Tensor,
            depth_hypos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Fitting probability curve: x is depth hypos, y is preb value, y=e**((x-depth)/s)
            Z = lny
            X = [[x1**2， 1]
                 [  ...    ]
                 [xn**2， 1]]
            B = (X.T.X)-1.X.T.Z     [b0, b1]
            s = -1/b0
            @param prob_volume:(B, D, H, W)
            @param depth_hypos: (B, D, H, W)|(B, D, 1, 1)
            """
        with torch.no_grad():
            B, D, H, W = prob_volume.shape
            if depth_hypos.shape[-1] != prob_volume.shape[-1]:
                depth_hypos = depth_hypos.view(B, D, 1, 1).repeat(1, 1, H, W)

            # Z
            prob_volume = prob_volume.clamp(min=1e-40)  # very key
            Z = torch.log(prob_volume).unsqueeze(-1).permute(0, 2, 3, 1, 4)  # (B, D, H, W, 1) -> (B, H, W, D, 1)

            # X
            X0 = torch.ones_like(depth_hypos)
            X1 = (depth_hypos - depth.unsqueeze(1)) ** 2
            X = torch.stack([X1, X0], dim=-1).permute(0, 2, 3, 1, 4)  # (B, D, H, W, 2) -> (B, H, W, D, 2)
            X_T = X.transpose(-1, -2)

            # B
            X = torch.matmul(torch.inverse(torch.matmul(X_T, X)), X_T)
            B = torch.matmul(X, Z).squeeze(-1)  # (B, H, W, 3)

            # s , u: likely <0 -> abs
            s = torch.abs(-1 / B[:, :, :, 0])  # (B, H, W)

            return s


    def _gauss_fitting1(
                self,
                depth: torch.Tensor,
                prob_volume: torch.Tensor,
                depth_hypos: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
          Fitting probability curve: x is depth hypos, y is preb value,  y=e**((x-u)/s)
          Z = lny
          X = [[1, x1, x1**2]
               [    ...   ]
               [1, xn, xn**2]]
          B = (X.T.X)-1.X.T.Z     [b0, b1, b2]
          s = -1/b0
          u = (s.b1)/2
          use: res = torch.sqrt(-1*s*torch.log(prob_values[n]))
          @param prob_volume:(B, D, H, W)
          @param depth_hypos: (B, D, H, W)|(B, D, 1, 1)
          """
        with torch.no_grad():
            B, D, H, W = prob_volume.shape
            if depth_hypos.shape[-1] != prob_volume.shape[-1]:
                depth_hypos = depth_hypos.view(B, D, 1, 1).repeat(1, 1, H, W)

            # Z
            prob_volume = prob_volume.clamp(min=1e-40)  # the key
            # print("prob",prob_volume.min(), prob_volume.max(), prob_volume.mean())
            Z = torch.log(prob_volume).unsqueeze(-1).permute(0, 2, 3, 1, 4)  # (B, D, H, W, 1) -> (B, H, W, D, 1)
            # print("Z",Z.min(), Z.max(), Z.mean())
            # X
            X0 = torch.ones_like(depth_hypos)
            X1 = depth_hypos
            X2 = depth_hypos ** 2
            X = torch.stack([X2, X1, X0], dim=-1).permute(0, 2, 3, 1, 4)  # (B, D, H, W, 3) -> (B, H, W, D, 3)
            X_T = X.transpose(-1, -2)

            # B
            B = torch.matmul(X_T, X)
            B = torch.inverse(B)
            B = torch.matmul(B, X_T)
            B = torch.matmul(B, Z).squeeze(-1)  # (B, H, W, 3)

            # s , u: likely <0 -> abs
            s = torch.abs(-1 / B[:, :, :, 0])  # (B, H, W)
            u = torch.abs(0.5 * B[:, :, :, 1] * s)  # (B, H, W)

            return s


def atv_hypos(depth, exp_variance, depth_range, ndepths, eps = 1e-12):

    B = depth_range.shape[0]
    # init hypos
    if depth is None:
        depth_min, depth_max = depth_range[:, 0].float(), depth_range[:, 1].float()
        depth_min, depth_max = depth_min.view(B, 1, 1, 1), depth_max.view(B, 1, 1, 1)
        # depth_min_inverse = 1.0 / depth_min
        # depth_max_inverse = 1.0 / depth_max
        # depth_sample = torch.arange(0, ndepths, step=1, device=depth_range.device).view(1, ndepths, 1, 1)
        # depth_hypos = depth_max_inverse + depth_sample / (ndepths - 1) * (
        #             depth_min_inverse - depth_max_inverse)
        # depth_hypos = torch.flip(1.0 / depth_hypos, dims=[1])
        depth_interval = (depth_max - depth_min) / (ndepths - 1)
        depth_hypos = depth_min.unsqueeze(1) \
                      + (torch.arange(0, ndepths, device=depth_range.device).reshape(1, -1)
                         * depth_interval.unsqueeze(1))

        return depth_hypos.view(B, ndepths, 1, 1)

    # depth = F.interpolate(depth.unsqueeze(1), scale_factor=2, mode='bilinear')
    depth = depth.detach()
    exp_variance = exp_variance.detach()
    exp_variance = F.interpolate(exp_variance.unsqueeze(1), scale_factor=2, mode='bilinear')

    low_bound = -torch.min(depth, exp_variance)
    high_bound = exp_variance
    step = (high_bound - low_bound) / (ndepths - 1)

    depth_hypos = []
    for i in range(ndepths):
        depth_hypos.append(depth + low_bound + step * i + eps)

    depth_hypos = torch.cat(depth_hypos, 1)
    # assert depth_range_samples.min() >= 0, depth_range_samples.min()
    return depth_hypos
