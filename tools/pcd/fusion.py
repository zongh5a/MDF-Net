import argparse
import torch, time
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from tqdm import tqdm
import numpy as np
import open3d as o3d
import os
import cv2

from utils.utils import (
    get_pixel_grids, 
    recursive_apply,
    unflatten,
    load_cam, load_pfm, load_pair,
    normalize_for_grid_sample, get_in_range,
    idx_cam2img, idx_cam2world, idx_img2cam, idx_img2world, idx_world2cam, idx_world2img,
)

_ext = load(name='fusion', sources=['utils/fusion.cpp'], extra_cflags=['-std=c++17', '-O3'])


def prob_filter(ref_probs, pthresh, greater=True):  # n3hw -> n1hw
    cmpr = lambda x, y: x > y if greater else lambda x, y: x < y
    masks = cmpr(ref_probs, torch.Tensor(pthresh).to(ref_probs.dtype).to(ref_probs.device).view(1,3,1,1)).to(ref_probs.dtype)
    mask = (masks.sum(dim=1, keepdim=True) >= (len(pthresh)-0.1))
    return mask

def get_reproj(ref_depth, srcs_depth, ref_cam, srcs_cam):  # 11hw, 1v1hw -> 1v3hw, 1v1hw
    _, v, _, h, w = srcs_depth.size()
    # proj ref to space
    ref_valid = ref_depth > 1e-9  # 11hw
    v_xy1_img = get_pixel_grids(h, w).reshape(1,1,h,w,3)[ref_valid].reshape(1,1,-1,3,1)  # 11m31
    ref_v_d = ref_depth[ref_valid].reshape(1,1,1,-1)  # 111m
    ref2_v_xyz1_wld = idx_img2world(v_xy1_img, ref_v_d, ref_cam)  # 11m41
    # proj space to src
    srcs_cam_f = srcs_cam[0]  # v244
    ref2src_v_xy1_img = idx_world2img(ref2_v_xyz1_wld, srcs_cam_f)  # v1m31
    # gather src depth
    grid = normalize_for_grid_sample(srcs_depth[0], ref2src_v_xy1_img[...,:2,0])  # v1m2
    grid_sample_in_range = get_in_range(grid)  # v1m
    gather_src_depth = F.grid_sample(srcs_depth[0], grid, 'nearest', 'zeros', False)  # v11m
    grid_sample_in_range *= (gather_src_depth > 1e-9).to(gather_src_depth.dtype).reshape(v,1,-1)  # v1m
    # back proj to ref
    ref2src2_v_xyz1_wld = idx_img2world(ref2src_v_xy1_img, gather_src_depth, srcs_cam_f)  # v1m41
    ref2src2ref_v_xyz1_cam = idx_world2cam(ref2src2_v_xyz1_wld, ref_cam)  # v1m41
    ref2src2ref_v_xy1_img = idx_cam2img(ref2src2ref_v_xyz1_cam, ref_cam)  # v1m31
    reproj_v_xydm = torch.cat([ref2src2ref_v_xy1_img[...,:2,0], ref2src2ref_v_xyz1_cam[...,2:3,0], grid_sample_in_range.unsqueeze(-1)], -1)  # v1m4
    # unflatten
    reproj_xydm = unflatten(reproj_v_xydm[:,0].permute(1,2,0), ref_valid[0,0])  # hw4v
    reproj_xydm = reproj_xydm.permute(3,2,0,1).unsqueeze(0) # 1v4hw
    reproj_xyd = reproj_xydm[:,:,:3,:,:]  # 1v3hw
    in_range = reproj_xydm[:,:,3:,:,:] * ref_valid.unsqueeze(1)  # 1v1hw
    return reproj_xyd, in_range

def vis_filter(ref_depth, reproj_xyd, in_range, img_dist_thresh, depth_thresh, vthresh):
    n, v, _, h, w = reproj_xyd.size()
    xy = get_pixel_grids(h, w).permute(3,2,0,1).unsqueeze(1)[:,:,:2]  # 112hw
    dist_masks = (reproj_xyd[:,:,:2,:,:] - xy).norm(dim=2, keepdim=True) < img_dist_thresh  # nv1hw
    depth_masks = (ref_depth.unsqueeze(1) - reproj_xyd[:,:,2:,:,:]).abs() < (torch.max(ref_depth.unsqueeze(1), reproj_xyd[:,:,2:,:,:])*depth_thresh)  # nv1hw
    masks = in_range * dist_masks.to(ref_depth.dtype) * depth_masks.to(ref_depth.dtype)  # nv1hw
    mask = masks.sum(dim=1) >= (vthresh-1.1)  # n1hw
    return masks, mask

def ave_fusion(ref_depth, reproj_xyd, masks):
    ave = ((reproj_xyd[:,:,2:,:,:]*masks).sum(dim=1)+ref_depth) / (masks.sum(dim=1)+1)  # n1hw
    return ave

def vis_fusion(ref_depth, srcs_depth, ref_cam, srcs_cam):  # 11hw, 1v1hw -> 11hw
    _, v, _, h, w = srcs_depth.size()
    # ref pixels
    ref_valid = ref_depth > 1e-9  # 11hw
    ref_v_d = ref_depth[ref_valid].reshape(1,1,1,-1)  # 111m
    ref_v_xy1_img = get_pixel_grids(h, w).reshape(1,1,h,w,3)[ref_valid].reshape(1,1,-1,3,1)  # 11m31
    # src pixels (proj to ref)
    srcs_depth_f = srcs_depth[0]  # v1hw
    srcs_cam_f = srcs_cam[0]  # v244
    srcs_valid = srcs_depth_f > 1e-9  # v1hw
    src_xy1_img = get_pixel_grids(h, w).reshape(1,h,w,3,1)  # 1hw31
    src2_xyz1_wld = idx_img2world(src_xy1_img, srcs_depth_f, srcs_cam_f)  # vhw41
    src2_v_xyz1_wld = src2_xyz1_wld[srcs_valid[:,0]].reshape(1,1,-1,4,1)  # 11M41
    src2ref_v_xyz1_cam = idx_world2cam(src2_v_xyz1_wld, ref_cam)  # 11M41
    src2ref_v_xy1_img = idx_cam2img(src2ref_v_xyz1_cam, ref_cam)  # 11M31
    all_xy1_img = torch.cat([ref_v_xy1_img, src2ref_v_xy1_img], 2)  # 11M31
    all_d = torch.cat([ref_v_d, src2ref_v_xyz1_cam[...,2:3,0].reshape(1,1,1,-1)], 3)  # 111M
    all2_xyz1_wld = idx_img2world(all_xy1_img, all_d, ref_cam)  # 11M41
    violation = []
    for all2_xyz1_wld_chunk in all2_xyz1_wld.split(500000, dim=2):
        all2src_xyz1_cam = idx_world2cam(all2_xyz1_wld_chunk, srcs_cam_f)  # v1M41
        all2src_xy1_img = idx_cam2img(all2src_xyz1_cam, srcs_cam_f)  # v1M31
        grid = normalize_for_grid_sample(srcs_depth_f, all2src_xy1_img[...,:2,0])  # v1M2
        # grid_sample_in_range = get_in_range(grid)  # v1M
        gather_src_depth = F.grid_sample(srcs_depth_f, grid, 'nearest', 'zeros', False)  # v11M
        is_violation = gather_src_depth > all2src_xyz1_cam[...,2,0].reshape(v,1,1,-1)  # v11M
        violation.append(is_violation.int().sum(0).reshape(-1).int().cpu())  # M
    violation = torch.cat(violation)
    out = _ext.vis_fusion_core(all_d.reshape(-1).cpu(), all_xy1_img[0,0,:,:2,0].cpu(), violation, ref_valid[0,0].cpu()).reshape(1,1,h,w)
    return out

def small_seg_filter(depth, window_size, diff_thresh, size_thresh):
    return _ext.small_seg_core(depth[0,0], window_size, diff_thresh, size_thresh).bool()

def batch_vis_filter(views, pair, args):
    update = {}
    for i, id in tqdm(enumerate(pair['id_list']), 'vis filter', len(pair['id_list'])):
        srcs_id = [loop_id for loop_id in pair[id]['pair'] if loop_id in pair][:args.view]
        ref_depth_g, ref_cam_g = views[id]['depth'].cuda(), views[id]['cam'].cuda()
        srcs_depth_g, srcs_cam_g = [torch.stack([views[loop_id][attr] for loop_id in srcs_id], dim=1).cuda() for attr in ['depth', 'cam']]

        reproj_xyd_g, in_range_g = get_reproj(ref_depth_g, srcs_depth_g, ref_cam_g, srcs_cam_g)
        vis_masks_g, vis_mask_g = vis_filter(ref_depth_g, reproj_xyd_g, in_range_g, 1, 0.01, args.vthresh)

        update[id] = {
            'mask': vis_mask_g.cpu()
        }
    for i, id in enumerate(pair['id_list']):
        views[id]['mask'] = views[id]['mask'] & update[id]['mask']
        views[id]['depth'] *= views[id]['mask']

def get_cloud(dataset_root, scan, img_folder, cam_folder,
           eval_folder, photo_threshold, args):
    dataset_root, scan, img_folder, cam_folder,
    scan_location = os.path.join(dataset_root, scan)
    eval_location = os.path.join(eval_folder, scan)

    pair_path = os.path.join(scan_location, "pair.txt")

    pair = load_pair(pair_path)  # , min_views=num_src)
    n_views = len(pair['id_list'])

    views = {}

    for i, id in tqdm(enumerate(pair['id_list']), 'load data', n_views):
        # image = cv2.imread(f'{args.data}/{id.zfill(8)}.jpg').transpose(2, 0, 1)[::-1]
        # cam = load_cam(f'{args.data}/cam_{id.zfill(8)}_flow3.txt', 256, 1)
        # depth = np.expand_dims(load_pfm(f'{args.data}/{id.zfill(8)}_flow3.pfm'), axis=0)
        # probs = np.stack([load_pfm(f'{args.data}/{id.zfill(8)}_flow{k + 1}_prob.pfm') for k in range(3)], axis=0)

        img_path = os.path.join(scan_location, img_folder, '{:0>8}.jpg'.format(id))
        cam_path = os.path.join(scan_location, cam_folder, '{:0>8}_cam.txt'.format(id))
        depth_est_path = os.path.join(eval_location, "depth_est",'{:0>8}'.format(id) + '.pfm')
        confidence_path = os.path.join(eval_location, "confidence",'{:0>8}'.format(id) + '.pfm')

        image = cv2.imread(img_path).transpose(2, 0, 1)[::-1]
        cam = load_cam(cam_path, 256, 1)
        depth = np.expand_dims(load_pfm(depth_est_path), axis=0)
        shape = depth.shape

        # probs = np.stack([load_pfm(confidence_path) for k in range(3)], axis=0)
        probs = np.stack([load_pfm(confidence_path)], axis=0)
        views[id] = {
            'image': image[:, :shape[1], :shape[2]],  # 13hw (after next step)
            'cam': cam,  # 1244
            'depth': depth,  # 11hw
            'prob': probs,  # 13hw
        }
        recursive_apply(views[id], lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0))

    for i, id in tqdm(enumerate(pair['id_list']), 'prob filter', n_views):
        # views[id]['mask'] = prob_filter(views[id]['prob'].cuda(), photo_threshold).cpu()  # 11hw bool
        views[id]['mask'] = (views[id]['prob'].cuda() > photo_threshold).cpu()
        views[id]['depth'] *= views[id]['mask']

    batch_vis_filter(views, pair, args)

    update = {}
    for i, id in tqdm(enumerate(pair['id_list']), 'vis fusion', n_views):
        srcs_id = [loop_id for loop_id in pair[id]['pair'] if loop_id in pair][:args.view]
        ref_depth_g, ref_cam_g = views[id]['depth'].cuda(), views[id]['cam'].cuda()
        srcs_depth_g, srcs_cam_g = [torch.stack([views[loop_id][attr] for loop_id in srcs_id], dim=1).cuda() for attr in
                                    ['depth', 'cam']]

        ref_depth_vis = vis_fusion(ref_depth_g, srcs_depth_g, ref_cam_g, srcs_cam_g)

        update[id] = {
            'depth': ref_depth_vis
        }
        del ref_depth_g, ref_cam_g, srcs_depth_g, srcs_cam_g
    for i, id in enumerate(pair['id_list']):
        views[id]['depth'] = update[id]['depth'] * views[id]['mask']

    batch_vis_filter(views, pair, args)

    update = {}
    for i, id in tqdm(enumerate(pair['id_list']), 'ave fusion', n_views):
        srcs_id = [loop_id for loop_id in pair[id]['pair'] if loop_id in pair][:args.view]
        ref_depth_g, ref_cam_g = views[id]['depth'].cuda(), views[id]['cam'].cuda()
        srcs_depth_g, srcs_cam_g = [torch.stack([views[loop_id][attr] for loop_id in srcs_id], dim=1).cuda() for attr in
                                    ['depth', 'cam']]

        reproj_xyd_g, in_range_g = get_reproj(ref_depth_g, srcs_depth_g, ref_cam_g, srcs_cam_g)
        vis_masks_g, vis_mask_g = vis_filter(ref_depth_g, reproj_xyd_g, in_range_g, 1, 0.01, args.vthresh)

        ref_depth_ave_g = ave_fusion(ref_depth_g, reproj_xyd_g, vis_masks_g)

        update[id] = {
            'depth': ref_depth_ave_g.cpu(),
            # 'mask': vis_mask_g.cpu()
        }
        del ref_depth_g, ref_cam_g, srcs_depth_g, srcs_cam_g, reproj_xyd_g, in_range_g, vis_masks_g, vis_mask_g, ref_depth_ave_g
    for i, id in enumerate(pair['id_list']):
        # views[id]['mask'] = views[id]['mask'] & update[id]['mask']
        views[id]['depth'] = update[id]['depth'] * views[id]['mask']

    batch_vis_filter(views, pair, args)

    update = {}
    for i, id in tqdm(enumerate(pair['id_list']), 'small seg filter', n_views):
        small_seg_mask = small_seg_filter(views[id]['depth'], 4, 1e-3, 10)
        update[id] = {'mask': small_seg_mask}
    for i, id in enumerate(pair['id_list']):
        views[id]['mask'] = views[id]['mask'] & update[id]['mask']
        views[id]['depth'] *= views[id]['mask']

    if args.write_mask:
        for i, id in tqdm(enumerate(pair['id_list']), 'write masks', n_views):
            cv2.imwrite(f'{eval_location}/{args.filter_folder}/{id.zfill(8)}_mask.png', views[id]['mask'][0, 0].numpy().astype(np.uint8) * 255)

    pcds = {}
    for i, id in tqdm(enumerate(pair['id_list']), 'back proj', n_views):
        ref_depth_g, ref_cam_g = views[id]['depth'].cuda(), views[id]['cam'].cuda()

        idx_img_g = get_pixel_grids(*ref_depth_g.size()[-2:]).unsqueeze(0)
        idx_cam_g = idx_img2cam(idx_img_g, ref_depth_g, ref_cam_g)
        points_g = idx_cam2world(idx_cam_g, ref_cam_g)[..., :3, 0]  # nhw3
        cam_center_g = (- ref_cam_g[:, 0, :3, :3].transpose(-2, -1) @ ref_cam_g[:, 0, :3, 3:])[..., 0]  # n3
        dir_vec_g = cam_center_g.reshape(-1, 1, 1, 3) - points_g  # nhw3

        p_f = points_g.cpu()[views[id]['mask'].squeeze(1)]  # m3
        c_f = views[id]['image'].permute(0, 2, 3, 1)[views[id]['mask'].squeeze(1)] / 255  # m3
        d_f = dir_vec_g.cpu()[views[id]['mask'].squeeze(1)]  # m3

        pcds[id] = {
            'points': p_f,
            'colors': c_f,
            'dirs': d_f,
        }
        del views[id]

    print('Construct combined PCD')
    all_points, all_colors, all_dirs = \
        [torch.cat([pcds[id][attr] for id in pair['id_list']], dim=0) for attr in ['points', 'colors', 'dirs']]
    print(f'Number of points {all_points.shape[0]}')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(all_colors.numpy())

    if not args.no_normal:
        print('Estimate normal')
        pcd.estimate_normals()
        all_normals_np = np.asarray(pcd.normals)
        is_same_dir = (all_normals_np * all_dirs.numpy()).sum(-1, keepdims=True) > 0
        all_normals_np *= is_same_dir.astype(np.float32) * 2 - 1
        pcd.normals = o3d.utility.Vector3dVector(all_normals_np)

    if args.downsample is not None:
        print('Down sample')
        if args.downsample == -1:
            nn_dist = np.asarray(pcd.compute_nearest_neighbor_distance())
            den = np.percentile(nn_dist, 90)
        else:
            den = args.downsample
        pcd = pcd.voxel_down_sample(den)

    if args.outply_folder is None:
        ply_path = os.path.join(eval_location, args.filter_folder, scan+'.ply')
    else:
        ply_path = os.path.join(args.outply_folder, scan+'.ply')

    o3d.io.write_point_cloud(ply_path, pcd, print_progress=True)
    print("save in:" + ply_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--view', type=int, default=10)
    parser.add_argument('--vthresh', type=int, default=4)
    parser.add_argument('--cam_scale', type=float, default=1)
    # parser.add_argument('--show_result', action='store_true', default=False)
    parser.add_argument('--downsample', type=float, default=None)
    parser.add_argument('--no_normal', action='store_true', default=False)
    parser.add_argument('--write_mask', action='store_true', default=False)

    parser.add_argument("-r", '--root_folder', default="/hy-tmp", type=str, help='dataset root location')
    parser.add_argument("-e", '--eval_folder', default="/hy-tmp/outputs", type=str, help='eval output location')  # "../../outputs"
    parser.add_argument('-o', '--outply_folder', default="/hy-tmp/oursply", type=str, help='')
    parser.add_argument('-d', '--dataset', default='tanks', type=str, help='dtu or tanks')
    parser.add_argument('-s', '--set', default='intermediate', type=str, help='tanks set:intermediate or advanced')
    parser.add_argument('-f', '--filter_folder', default="filter", type=str, help='filter output location')

    args = parser.parse_args()

    # pthresh = [float(v) for v in args.pthresh.split(',')]
    num_src = args.view
    photo_threshold = 0.8

    root_dir = args.root_folder
    scans, dataset_root, img_folder, cam_folder, nconditions = None, None, None, None, None

    if args.dataset == "dtu":
        dataset_root = os.path.join(root_dir, "dtu1600x1200")
        # dtu_labels = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
        dtu_labels = [11,]# 48]
        scans = ["scan"+str(label) for label in dtu_labels]
        #nviewss = [49,]*len(dtu_labels)
        img_folder = "images"
        cam_folder = "cams"

    elif args.dataset == "tanks":
        tanks_set = args.set  # "intermediate"
        dataset_root = os.path.join(root_dir, "TankandTemples", tanks_set, )
        if tanks_set == "intermediate":
            scans = ['Family','Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']#condition>5
            # scans = ['Horse',]

        elif tanks_set == "advanced":
            scans = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']#condition>5
            # scans = ['Auditorium', 'Courtroom',]#condition>3

        img_folder =  "images"
        cam_folder = "cams_1"

    else:
        print("please use dtu or tanks dataset,exit!")
        exit()

    for scan in scans:
        print("scan:", scan)
        start_time = time.time()

        get_cloud(dataset_root, scan, img_folder, cam_folder,
           args.eval_folder, photo_threshold, args)

        print("all time:", (time.time()-start_time)/60, "min")

