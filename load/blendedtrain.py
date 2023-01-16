import cv2, random, os
import torch
import numpy as np

from load.getpath import get_img_path, get_cam_path, get_depth_path
from tools import data_io


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self,
                 datasetpath: str,
                 nviews: int=5,
                 robust_train: bool=False,
                 ) -> None:
        super(LoadDataset, self).__init__()
        self.datasetpath = datasetpath
        self.listfile = os.path.join(datasetpath, "training_list.txt")
        self.nviews = nviews
        self.robust_train = robust_train

        self.all_compose = self._copmose_input()

    def __getitem__(self, item):
        scan, ref_view, src_views = self.all_compose[item]
        rs_views = [ref_view] + src_views[:self.nviews - 1]

        if self.robust_train:
            src_views = src_views[:7]
            index = random.sample(range(1, len(src_views), 1), self.nviews - 1)
            rs_views = [ref_view] + [src_views[i] for i in index]

        imgs, ref_depths, extrinsics, intrinsics = [], {}, [], []
        depth_min, depth_max, depth_range = 0.0, 0.0, None
        for i, vid in enumerate(rs_views):
            img_filename = get_img_path(self.datasetpath, scan, vid, mode="blendedmvs")
            cam_filename = get_cam_path(self.datasetpath, scan, vid, mode="blendedmvs")

            imgs.append(data_io.read_img(img_filename))
            intrinsic, extrinsic, depth_min, depth_max = self._read_cam_file(cam_filename) # (3,3) (4,4)

            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)

            if i == 0:  # reference view
                depth_filename = get_depth_path(self.datasetpath, scan, vid, mode="blendedmvs")
                ref_depth = np.array(data_io.read_pfm(depth_filename)[0], dtype=np.float32)
                h, w = ref_depth.shape
                ref_depths["3"] = cv2.resize(ref_depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST)
                ref_depths["2"] = cv2.resize(ref_depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
                ref_depths["1"] = cv2.resize(ref_depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
                ref_depths["0"] = ref_depth

                depth_range = np.array([depth_min, depth_max])

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        return {"imgs": imgs,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "ref_depths": ref_depths,
                "depth_range": depth_range
                }

    def __len__(self):
        return len(self.all_compose)

    def _copmose_input(self):
        all_compose = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)

            with open(os.path.join(self.datasetpath, pair_file)) as f:
                num_viewpoint = int(f.readline())

                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        all_compose.append((scan, ref_view, src_views))

        return all_compose

    def _read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[3])

        return intrinsics, extrinsics, depth_min, depth_max


if __name__=="__main__":
    dataset = LoadDataset(datasetpath=".")
    # print(dataset.)
