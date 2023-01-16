import torch, os
import numpy as np
from typing import List

from tools import data_io
from load.getpath import get_img_path,get_cam_path


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self,
                 datasetpath: str,
                 scenelist: List,
                 nviews: int,
                 ) -> None:
        super(LoadDataset, self).__init__()
        self.datasetpath = datasetpath
        self.nviews = nviews
        self.all_compose = []
        for scan in scenelist:
            num_viewpoint, pair_data = data_io.read_pairfile(os.path.join(self.datasetpath, scan, 'pair.txt'))
            for ref, srcs in pair_data:
                self.all_compose.append([scan, ref, srcs])


    def __getitem__(self, item):
        scene,ref_view, src_views = self.all_compose[item]
        rs_views = [ref_view] + src_views[:self.nviews - 1]

        imgs,  extrinsics, intrinsics ,depth_ranges = [], [], [], []
        scan_folder = scene

        for vid in rs_views:
            img_filename = get_img_path(self.datasetpath, scan_folder, vid, mode="tanks")
            cam_filename = get_cam_path(self.datasetpath, scan_folder, vid, mode="tanks")

            imgs.append(data_io.read_img(img_filename)[:1056]) #(2048/1920,1080)->(2048/1920,1056)  1024
            intrinsic, extrinsic, depth_range = self._read_cam_file(cam_filename)

            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            depth_ranges.append(depth_range)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        return {"imgs": imgs,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "depth_range": depth_ranges[0],  # use ref depth_range
                "filename":scene + '/{}/' + '{:0>8}'.format(ref_view) + "{}"
                }

    def __len__(self):
        # nscans* nviews（49）* nlightings（7）
        return len(self.all_compose)

    def _read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsic = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsic = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        depth_range = np.fromstring(lines[11], dtype=np.float32, sep=' ')

        return intrinsic, extrinsic, depth_range


if __name__=="__main__":
    pass
