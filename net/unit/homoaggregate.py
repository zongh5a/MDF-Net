import torch
import torch.nn as nn
import torch.nn.functional as F

from net.unit.base import ConvBNReLU3D, homo_warping


class VectorAggregate(nn.Module):
    def __init__(self,
                 ngroups: int=8,
                 ):
        super(VectorAggregate, self).__init__()
        self.ngroups = ngroups

        # (B, C, D, H, W) -> (B, 1, D, H, W)
        self.depth_weight = nn.Sequential(
            ConvBNReLU3D(ngroups, 1, 1, 1, 0),
            nn.Conv3d(1, 1, 1, 1, 0, ),
            nn.Sigmoid(),
        )

        print('{} parameters: {}'
              .format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self, features, ref_proj, src_projs, depth_hypos, ):
        D = depth_hypos.shape[1]
        ref_feature, src_features = features[0], features[1:]  # (B,C,H,W),(nviews-1)*（B,C,H,W）

        B, C, H, W = ref_feature.shape
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)  # （B,C,D,H,W）
        # convert to unit vector
        ref_volume = F.softmax(ref_volume.view(B, self.ngroups, C // self.ngroups, D, H, W), dim=2)

        volume_sum, weight_sum = 0.0, 0.0
        for src_fea, src_proj in zip(src_features, src_projs):
            # torch.cuda.empty_cache()
            volume = homo_warping(src_fea, src_proj, ref_proj, depth_hypos)
            volume = F.softmax(volume.view(B, self.ngroups, C // self.ngroups, D, H, W), dim=2)
            volume = torch.sum(volume * ref_volume, dim=2)
            weight = self.depth_weight(volume)
            weight_sum += weight
            volume_sum += weight * volume

            del volume, weight

        return volume_sum / weight_sum


def homo_aggregate_by_variance(features, ref_proj, src_projs, depth_hypos):

    ndepths = depth_hypos.shape[1]
    ref_feature, src_features = features[0], features[1:]  # (B,C,H,W),(nviews-1)*（B,C,H,W）
    # ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, ndepths, 1, 1)  # （B,C,D,H,W）

    ref_feature = ref_feature.unsqueeze(2)
    volume_sum, volume_sq_sum = ref_feature, ref_feature**2
    for src_fea, src_proj in zip(src_features, src_projs):
        # torch.cuda.empty_cache()
        warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_hypos)
        warped_volume = F.softmax(warped_volume, dim=1)

        volume_sum = volume_sum + warped_volume
        volume_sq_sum = volume_sq_sum + warped_volume ** 2
        del warped_volume

    cost_volume = volume_sq_sum.div_(len(src_features)+1).sub_(volume_sum.div_(len(src_features)+1).pow_(2))
    del volume_sum, volume_sq_sum

    return cost_volume

# def homo_aggregate_by_variance(features, ref_proj, src_projs, depth_hypos):
#
#     ndepths = depth_hypos.shape[1]
#     ref_feature, src_features = features[0], features[1:]  # (B,C,H,W),(nviews-1)*（B,C,H,W）
#     # ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, ndepths, 1, 1)  # （B,C,D,H,W）
#
#     volume_sum, volume_sq_sum = 0.0, 0.0
#
#     for src_fea, src_proj in zip(src_features, src_projs):
#         # torch.cuda.empty_cache()
#         warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_hypos)
#         volume_sum = volume_sum + warped_volume
#         volume_sq_sum = volume_sq_sum + warped_volume ** 2
#         del warped_volume
#
#     cost_volume = volume_sq_sum.div_(len(src_features)).sub_(volume_sum.div_(len(src_features)).pow_(2))
#     del volume_sum, volume_sq_sum
#
#     return cost_volume



if __name__=="__main__":
    pass


