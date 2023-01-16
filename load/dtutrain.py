import cv2, random
import torch
import numpy as np
from typing import List

from tools import data_io
from load.getpath import get_img_path,get_cam_path,get_depth_path


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self,
                 datasetpath: str,
                 pairpath: str,
                 scencelist: List,
                 lighting_label: List,
                 nviews: int,
                 robust_train: bool=False,
                 ) -> None:
        super(LoadDataset, self).__init__()
        self.datasetpath = datasetpath
        self.scenelist = scencelist
        self.lighting_label = lighting_label
        self.nviews = nviews
        self.robust_train = robust_train

        self.num_viewpoint,self.pairs = data_io.read_pairfile(pairpath)
        self.all_compose = self._copmose_input()

    def __getitem__(self, item):
        scene,lighting,ref_view, src_views = self.all_compose[item]
        rs_views = [ref_view] + src_views[:self.nviews - 1]

        if self.robust_train:
            index = random.sample(range(1, len(src_views), 1), self.nviews - 1)
            rs_views = [ref_view] + [src_views[i] for i in index]

        imgs, ref_depths, extrinsics, intrinsics = [], {}, [], []
        scan_folder = "scan{}_train".format(scene)

        for i, vid in enumerate(rs_views):

            img_filename = get_img_path(self.datasetpath, scan_folder, vid, lighting, mode="train")
            cam_filename = get_cam_path(self.datasetpath, scan_folder, vid, mode="train")

            imgs.append(data_io.read_img(img_filename))
            intrinsic, extrinsic = data_io.read_cam_file(cam_filename) # (3,3) (4,4)

            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)

            if i == 0:  # reference view
                depth_filename = get_depth_path(self.datasetpath, scan_folder, vid, mode="train")
                ref_depth = np.array(data_io.read_pfm(depth_filename)[0], dtype=np.float32)
                h, w = ref_depth.shape
                ref_depths["3"] = cv2.resize(ref_depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST)
                ref_depths["2"] = cv2.resize(ref_depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
                ref_depths["1"] = cv2.resize(ref_depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
                ref_depths["0"] = ref_depth

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        return {"imgs": imgs,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "ref_depths": ref_depths,
                "depth_range": np.array([425.0, 935.0])
                }

    def __len__(self):
        # nscans* nviews（49）* nlightings（7）
        return len(self.scenelist)*len(self.pairs)*len(self.lighting_label)

    def _copmose_input(self):
        all_compose = []
        for scene in self.scenelist:
            for r,s in self.pairs:
                for lighting in self.lighting_label:
                    all_compose.append([scene,lighting,r,s])

        return all_compose


if __name__=="__main__":
    pass
