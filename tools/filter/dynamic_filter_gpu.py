import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'#'0,1,2,3,4,5,6,7'

import argparse, time, torch
import numpy as np
from plyfile import PlyData, PlyElement

from data_io import read_pfm, save_pfm, read_pairfile, read_img, read_cam_file, save_mask, bilinear_sampler


def filter(dataset_root, scan, img_folder, cam_folder,
           eval_folder, filter_folder, outply_folder,
           photo_threshold=0.8, nconditions=5, thre1=4, thre2=1300.):
    """
    # 1. filtered with photo consistency and geo consistency
    # 2. map to world location,and save to ply
    @param dataset_root: input folder:scene,include camera and photo
    @param scan:
    @param img_folder:
    @param cam_folder:
    @param eval_folder: include depth_est,photo consistency
    @param filter_folder: output folder:mask, depth_filter,
    @param photo_threshold:
    @param diff_threshold:
    """

    scan_location = os.path.join(dataset_root, scan)
    pair_path = os.path.join(scan_location, "pair.txt")

    eval_location = os.path.join(eval_folder, scan)

    filter_workspace = os.path.join(eval_location, filter_folder)
    os.makedirs(filter_workspace, exist_ok=True)

    vertexs, vertex_colors = [], []
    num_viewpoint, pairs = read_pairfile(pair_path)

    for ref_view, src_views in pairs:
        start_time = time.time()

        cam_path = os.path.join(scan_location, cam_folder,'{:0>8}_cam.txt'.format(ref_view))
        refimg_path = os.path.join(scan_location, img_folder,'{:0>8}.jpg'.format(ref_view))

        ref_intrinsics, ref_extrinsics = read_cam_file(cam_path)
        ref_img = read_img(refimg_path)

        depth_est_path = os.path.join(eval_location, "depth_est",'{:0>8}'.format(ref_view) + '.pfm')
        confidence_path = os.path.join(eval_location, "confidence",'{:0>8}'.format(ref_view) + '.pfm')

        ref_depth_est, scale = read_pfm(depth_est_path)
        confidence, _ = read_pfm(confidence_path)

        # to cuda
        ref_depth_est = torch.from_numpy(ref_depth_est.copy()).float().cuda()
        confidence = torch.from_numpy(confidence.copy()).float().cuda()
        ref_intrinsics = torch.from_numpy(ref_intrinsics.copy()).float().cuda()
        ref_extrinsics = torch.from_numpy(ref_extrinsics.copy()).float().cuda()

        h, w = confidence.shape

        # compute the geometric mask
        avg_mask, all_srcview_depth_ests, dynamic_mask_sum = 0, [], []
        for index, src_view in enumerate(src_views):

            src_cam_path=os.path.join(scan_location, cam_folder, '{:0>8}_cam.txt'.format(src_view))
            src_depth_est=os.path.join(eval_location, "depth_est", '{:0>8}'.format(src_view) + '.pfm')

            src_intrinsics, src_extrinsics = read_cam_file(src_cam_path)
            src_depth_est = read_pfm(src_depth_est)[0]

            src_depth_est = torch.from_numpy(src_depth_est.copy()).float().cuda()
            src_intrinsics = torch.from_numpy(src_intrinsics.copy()).float().cuda()
            src_extrinsics = torch.from_numpy(src_extrinsics.copy()).float().cuda()

            # check geometric consistency
            dynamic_masks, mask, depth_reprojected = check_geometric_consistency(ref_depth_est, ref_intrinsics,ref_extrinsics,
                                                                            src_depth_est,src_intrinsics, src_extrinsics,
                                                                            thre1, thre2)

            dynamic_masks = [m.float() for m in dynamic_masks]
            if index == 0:
                dynamic_mask_sum = dynamic_masks
            else:
                for i in range(9):
                    dynamic_mask_sum[i] += dynamic_masks[i]

            avg_mask += mask
            all_srcview_depth_ests.append(depth_reprojected)

        geo_mask = 0
        if len(dynamic_mask_sum) != 0:
            for i in range(2, 11):
                geo_mask += (dynamic_mask_sum[i - 2] >= i)  # iter result save in geo_mask
        else:
            continue

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (avg_mask + 1)

        # at least 3 source views matched
        geo_mask = geo_mask >= nconditions      # condition filter
        photo_mask = confidence > photo_threshold   # photo consistency
        final_mask = torch.logical_and(photo_mask, geo_mask)

        # to numpy
        depth_est_averaged = depth_est_averaged[0].cpu().numpy()
        geo_mask = geo_mask[0].cpu().numpy()
        photo_mask = photo_mask.cpu().numpy()
        final_mask = final_mask[0].cpu().numpy()
        ref_intrinsics = ref_intrinsics.cpu().numpy()
        ref_extrinsics = ref_extrinsics.cpu().numpy()
        ref_depth_est = ref_depth_est.cpu().numpy()


        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".
              format(scan_location, ref_view, photo_mask.sum(), geo_mask.sum(), final_mask.sum()),
              " time:",time.time()-start_time)

        # save mask
        save_mask(os.path.join(filter_workspace, "{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(filter_workspace, "{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(filter_workspace, "{:0>8}_final.png".format(ref_view)), final_mask)

        save_pfm(os.path.join(filter_workspace, "{}".format(ref_view)+"_" + "depth_est.pfm"),
                                ref_depth_est * final_mask.astype(np.float32))

        #######################################################################################

        # 2.map to world location,and save to ply
        height, width = depth_est_averaged.shape[:2]
        valid_points = final_mask

        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        color = ref_img[:h, :w, :][valid_points]

        # pix to camera ,to world
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    if outply_folder is None:
        PlyData([el]).write(os.path.join(eval_location, scan+".ply"))
        print("saving the final model to", os.path.join(eval_location, scan + ".ply"))
    else:
        os.makedirs(outply_folder, exist_ok=True)
        PlyData([el]).write(os.path.join(outply_folder, scan+".ply"))
        print("saving the final model to", os.path.join(outply_folder, scan + ".ply"))

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref,
                                depth_src, intrinsics_src, extrinsics_src,
                                thre1=4, thre2=1300.):
    height, width = depth_ref.shape
    batch = 1
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device),
                                  torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch, 1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch, 1, 1)
    inputs = [depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src]
    outputs = reproject_with_depth(*inputs)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = outputs
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    masks = []
    for i in range(2, 11):
        mask = torch.logical_and(dist < i / thre1, relative_depth_diff < i / thre2)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    height, width = depth_ref.shape
    batch = 1
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    x_ref, y_ref = x_ref.reshape(batch, -1), y_ref.reshape(batch, -1)
    # reference 3D space

    A = torch.inverse(intrinsics_ref)
    B = torch.stack((x_ref, y_ref, torch.ones_like(x_ref).to(x_ref.device)), dim=1) * depth_ref.reshape(batch, 1, -1)
    xyz_ref = torch.matmul(A, B)

    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                        torch.cat((xyz_ref, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].reshape([batch, height, width]).float()
    y_src = xy_src[:, 1].reshape([batch, height, width]).float()

    # print(x_src, y_src)
    sampled_depth_src = bilinear_sampler(depth_src.view(batch, 1, height, width), torch.stack((x_src, y_src), dim=-1).view(batch, height, width, 2))

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                        torch.cat((xy_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1) * sampled_depth_src.reshape(batch, 1, -1))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.inverse(extrinsics_src)),
                                torch.cat((xyz_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].reshape([batch, height, width]).float()
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
    x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='filter to maks cloudpoints...')
    parser.add_argument('-f','--filter_folder', default="filter", type=str, help='filter output location')
    parser.add_argument('-d', '--dataset', default='tanks', type=str, help='dtu or tanks')
    parser.add_argument('-s', '--set', default='intermediate', type=str, help='tanks set:intermediate or advanced')
    # parser.add_argument("-e",'--eval_folder', default="/data/user10/outputs", type=str, help='eval output location')#"../../outputs"
    # parser.add_argument("-r",'--root_folder', default="/data/user10", type=str, help='dataset root location')
    # parser.add_argument('-o', '--outply_folder', default="/data/user10/oursply", type=str, help='')
    parser.add_argument("-e", '--eval_folder', default="/hy-tmp/outputs", type=str, help='eval output location')  # "../../outputs"
    parser.add_argument("-r", '--root_folder', default="/hy-nas", type=str, help='dataset root location')
    parser.add_argument('-o', '--outply_folder', default="/hy-tmp/oursply", type=str, help='')

    args = parser.parse_args()
    print(args)

    root_dir = args.root_folder
    photo_thresh = 0.8
    thre1 = 4
    thre2 = 1300

    scans, dataset_root, img_folder, cam_folder, nconditions = None, None, None, None, None
    if args.dataset == "dtu":
        dataset_root = os.path.join(root_dir, "dtu1600x1200")
        # dtu_labels = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
        dtu_labels = [11,]# 48]
        scans = ["scan"+str(label) for label in dtu_labels]
        #nviewss = [49,]*len(dtu_labels)
        img_folder = "images"
        cam_folder = "cams"
        nconditions = 5

    elif args.dataset == "tanks":
        tanks_set = args.set  # "intermediate"
        dataset_root = os.path.join(root_dir, "TankandTemples", tanks_set, )
        if tanks_set == "intermediate":
            scans = ['Family','Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']#condition>5
            # scans = ['Horse',]
            nconditions = 5

        elif tanks_set == "advanced":
            scans = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']#condition>5
            # scans = ['Auditorium', 'Courtroom',]#condition>3
            nconditions = 1

        img_folder =  "images"
        cam_folder = "cams_1"

    else:
        print("please use dtu or tanks dataset,exit!")
        exit()

    # filter with photometric confidence maps and geometric constraints,then map to world position
    for scan in scans:
        print("scan:", scan)
        start_time = time.time()
        filter(dataset_root, scan, img_folder, cam_folder,
           args.eval_folder, args.filter_folder, args.outply_folder,
               photo_thresh, nconditions, thre1, thre2)

        print("all time:", time.time()-start_time)




