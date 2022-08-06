from typing import List, Union, Dict
import re
import numpy as np
import torch


def recursive_apply(obj: Union[List, Dict], func):
    if type(obj) == dict:
        for k in obj:
            if type(obj[k]) == list or type(obj[k]) == dict:
                recursive_apply(obj[k], func)
            else:
                obj[k] = func(obj[k])
    if type(obj) == list:
        for i in range(len(obj)):
            if type(obj[i]) == list or type(obj[i]) == dict:
                recursive_apply(obj[i], func)
            else:
                obj[i] = func(obj[i])

def load_pfm(file: str):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(br'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = data[::-1, ...]  # cv2.flip(data, 0)
    return data

def load_cam(file: str, max_d, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    with open(file) as f:
        words = f.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

def normalize_for_grid_sample(input_, grid):
    grid_ = grid.clone()
    for dim in range(grid_.size()[-1]):
        grid_[..., dim] /= input_.size()[-1-dim]
    grid_ = (grid_ * 2 - 1).clamp(-1.1, 1.1)
    return grid_

def bin_op_reduce(lst: List, func):
    result = lst[0]
    for i in range(1, len(lst)):
        result = func(result, lst[i])
    return result

def get_in_range(grid):  # after normalization, keepdim=False
    masks = []
    for dim in range(grid.size()[-1]):
        masks += [grid[..., dim]<=1, grid[..., dim]>=-1]
    in_range = bin_op_reduce(masks, torch.min).to(grid.dtype)
    return in_range

def get_pixel_grids(height, width):
    x_coord = (torch.arange(width, dtype=torch.float32).cuda() + 0.5).repeat(height, 1)
    y_coord = (torch.arange(height, dtype=torch.float32).cuda() + 0.5).repeat(width, 1).t()
    ones = torch.ones_like(x_coord)
    indices_grid = torch.stack([x_coord, y_coord, ones], dim=-1).unsqueeze(-1)  # hw31
    return indices_grid

def load_pair(file: str, min_views: int=None):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    for i in range(1, 1+2*n_cam, 2):
        pair = []
        score = []
        img_id = lines[i].strip()
        pair_str = lines[i+1].strip().split(' ')
        n_pair = int(pair_str[0])
        if min_views is not None and n_pair < min_views: continue
        for j in range(1, 1+2*n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j+1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i//2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs

def idx_img2cam(idx_img_homo, depth, cam):  # nhw31, n1hw -> nhw41
    idx_cam = cam[:,1:2,:3,:3].unsqueeze(1).inverse() @ idx_img_homo  # nhw31
    idx_cam = idx_cam / (idx_cam[...,-1:,:]+1e-9) * depth.permute(0,2,3,1).unsqueeze(4)  # nhw31
    idx_cam_homo = torch.cat([idx_cam, torch.ones_like(idx_cam[...,-1:,:])], dim=-2)  # nhw41
    # FIXME: out-of-range is 0,0,0,1, will have valid coordinate in world
    return idx_cam_homo

def idx_cam2world(idx_cam_homo, cam):  # nhw41 -> nhw41
    idx_world_homo =  cam[:,0:1,...].unsqueeze(1).inverse() @ idx_cam_homo  # nhw41
    idx_world_homo = idx_world_homo / (idx_world_homo[...,-1:,:]+1e-9)  # nhw41
    return idx_world_homo

def idx_img2world(idx_img_homo, depth, cam):  # nhw31, n1hw -> nhw41
    res_cam = idx_img2cam(idx_img_homo, depth, cam)
    res_world = idx_cam2world(res_cam, cam)
    return res_world

def idx_world2cam(idx_world_homo, cam):  # nhw41 -> nhw41
    idx_cam_homo = cam[:,0:1,...].unsqueeze(1) @ idx_world_homo  # nhw41
    idx_cam_homo = idx_cam_homo / (idx_cam_homo[...,-1:,:]+1e-9)   # nhw41
    return idx_cam_homo

def idx_cam2img(idx_cam_homo, cam):  # nhw41 -> nhw31
    idx_cam = idx_cam_homo[...,:3,:] / (idx_cam_homo[...,3:4,:]+1e-9)  # nhw31
    idx_img_homo = cam[:,1:2,:3,:3].unsqueeze(1) @ idx_cam  # nhw31
    idx_img_homo = idx_img_homo / (idx_img_homo[...,-1:,:]+1e-9)
    return idx_img_homo

def idx_world2img(idx_world_homo, cam): # nhw41 -> nhw31
    res_cam = idx_world2cam(idx_world_homo, cam)
    res_img = idx_cam2img(res_cam, cam)
    return res_img

def unflatten(flat, mask):
    shape = mask.shape + flat.shape[1:]
    res = torch.zeros(shape, dtype=flat.dtype, device=flat.device)
    res[mask] = flat
    return res
