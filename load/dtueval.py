import torch
import numpy as np
from typing import List
from tools import data_io
from load.getpath import get_img_path,get_cam_path


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self,
                 datasetpath: str,
                 pairpath: str,
                 scencelist: List,
                 nviews: int,
                 ) -> None:
        super(LoadDataset, self).__init__()
        self.datasetpath = datasetpath
        self.scenelist = scencelist
        self.nviews = nviews

        self.num_viewpoint,self.pairs = data_io.read_pairfile(pairpath)
        self.all_compose = self._copmose_input()

    def __getitem__(self, item):
        scene,ref_view, src_views = self.all_compose[item]
        rs_views = [ref_view] + src_views[:self.nviews - 1]

        imgs,  extrinsics, intrinsics = [], [], []
        scan_folder = "scan{}".format(scene)

        for vid in rs_views:
            img_filename = get_img_path(self.datasetpath, scan_folder, vid, mode="eval")
            cam_filename = get_cam_path(self.datasetpath, scan_folder, vid, mode="eval")

            imgs.append(data_io.read_img(img_filename)[:1184])  # 1184 1152
            intrinsic, extrinsic = data_io.read_cam_file(cam_filename)

            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        return {"imgs": imgs,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "depth_range": np.array([425.0, 935.0]),#935
                "filename":"scan{}".format(scene) + '/{}/' + '{:0>8}'.format(ref_view) + "{}"
                }

    def __len__(self):
        # nscans* nviews（49）
        return len(self.scenelist)*len(self.pairs)

    def _copmose_input(self):
        all_compose = []
        for scene in self.scenelist:
            for r,ss in self.pairs:
                all_compose.append([scene,r,ss])

        return all_compose


if __name__=="__main__":
    pass
