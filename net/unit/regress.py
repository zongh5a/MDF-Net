import torch
import torch.nn.functional as F


def depth_regression(prob_volume, depth_hypos):

    return torch.sum(prob_volume * depth_hypos, 1)

def confidence_regress(prob_volume, last_confidence=None, n=4, pad=(0, 0, 0, 0, 1, 2)):   # n=2, pad=(0, 0, 0, 0, 0, 1)

    with torch.no_grad():
        B, ndepths, H, W = prob_volume.shape
        prob_volume_sum4 = n * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=pad),
                                            (n, 1, 1), stride=1, padding=0).squeeze(1)
        index = torch.arange(ndepths, device=prob_volume.device, dtype=torch.float) \
            .view(1, ndepths, 1, 1).repeat(B, 1, 1, 1)
        depth_index = depth_regression(prob_volume, index).long()
        confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

    if last_confidence is not None:
        last_confidence = torch.nn.functional.interpolate(last_confidence.unsqueeze(1), size=None, scale_factor=2,
                                                mode='bicubic', align_corners=None).squeeze(1)
        confidence = 0.8*last_confidence + 0.2*confidence

    return confidence


if __name__=="__main__":
    pass