import torch


def scale_cam(intrinsics, extrinsics, stage):
    """
    zoom the cam matrixs in iter
    @param intrinsics: （B,VIEW,3,3）
    @param extrinsics:  (B,VIEW,4,4）
    @param level:  level = len(self.ndepths) - 1 - iteration  # 43210 10
    @return:
    """
    level = 3 - stage  # 321   !0
    # scale intrinsics matrix & making proj matrix
    intrinsics_iter, proj_matrix = intrinsics.clone(), extrinsics.clone()    # must clone!!!
    intrinsics_iter[:, :, :2, :] = intrinsics_iter[:, :, :2, :] / pow(2, level)
    proj_matrix[:, :, :3, :4] = torch.matmul(intrinsics_iter, proj_matrix[:, :, :3, :4])
    proj_matrix = torch.unbind(proj_matrix, 1)  # VIEW*(B,4,4)
    ref_proj, src_projs = proj_matrix[0], proj_matrix[1:]

    return ref_proj, src_projs


if __name__=="__main__":
    pass