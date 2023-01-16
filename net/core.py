import torch


class CoreNet(torch.nn.Module):
    def __init__(self, Backbone, Depth_hypos, scale,
                     Homoaggre, Regular, Regress, Refine):
        """

        @param Backbone: unit.backbone
        @param Depth_hypos: nn.ModuleList([unit.depthhypos...])
        @param scale: unit.scale
        @param Homoaggre: nn.ModuleList([unit.scale, unit.scale, unit.scale])
        @param Regular: nn.ModuleList([unit.regualr, unit.regualr, unit.regualr])
        @param Regress: list[unit.regress.depth_regress, unit.regress.confidence_regress]
        @param Refine: unit.refine
        @param ndepths: [48, 24, 8]
        """
        super(CoreNet, self).__init__()

        self.Backbone = Backbone
        self.Depth_hypos = Depth_hypos
        self.scale = scale
        self.Homoaggre = Homoaggre
        self.Regular = Regular
        self.Depth_regress, self.Confidence_regress = Regress
        self.Refine = Refine

        print('{} parameters: {}'.format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self, origin_imgs, extrinsics, intrinsics, depth_range):
        """
        predict depth
        @param origin_imgs: （B,VIEW,C,H,W） view0 is ref img
        @param extrinsics: （B,VIEW,4,4）
        @param intrinsics: （B,VIEW,3,3）
        @param depth_range: (B, 2) B*(depth_min, depth_max) dtu: [425.0, 935.0] tanks: [-, -]
        @return:
        """
        origin_imgs = torch.unbind(origin_imgs.float(), 1)  # VIEW*(B,C,H,W)

        # 0. feature extraction
        features = [self.Backbone(img) for img in origin_imgs] #views * 3 * fea

        depth, depth_hypos, prob_volume, depths = None, None, None, []
        for stage, (Depth_hypos, Homoaggre, Regular) in \
                enumerate(zip(self.Depth_hypos, self.Homoaggre, self.Regular)):

            # 1. get features
            feature = [fea[stage] for fea in features]

            # 2.scale intrinsic matrix & cal proj matrix
            ref_proj, src_projs = self.scale(intrinsics, extrinsics, stage)

            # 3. depth hypos
            depth_hypos= Depth_hypos(depth, depth_range, prob_volume, depth_hypos, upsample =True)

            # 4.homo & aggrate
            cost_volume = Homoaggre(feature, ref_proj, src_projs, depth_hypos)

            # 5.regular
            prob_volume = Regular(cost_volume)  # (B,D,H,W)

            # 6.depth regress
            depth = self.Depth_regress(prob_volume, depth_hypos)
            depths.append(depth)

        # 8. confidence, upsample depth
        # depth = self.Refine(origin_imgs[0], depth, depth_range)
        depth = self.Refine(depth, depth_range)
        depths.append(depth)

        if self.training:
            return {"depth": depths, }

        confidence = self.Confidence_regress(prob_volume)
        confidence = torch.nn.functional.interpolate(confidence.unsqueeze(1), size=None, scale_factor=2,
                                                mode='nearest', align_corners=None).squeeze(1)
        return {"depth": depth, "confidence": confidence}


if __name__=="__main__":
    pass

