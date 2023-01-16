import conf
import logging
import os,shutil,argparse,cv2
from tool import *


def probability_filter(depth_folder, prob_folder, nviews, prob_threshold, depthfiltered_folder):

    for view in range(nviews):
        init_depth_map_path = os.path.join(depth_folder,"{:08d}.pfm".format(view))
        prob_map_path = os.path.join(prob_folder, "{:08d}.pfm".format(view))
        out_depth_map_path = os.path.join(depthfiltered_folder, "{:08d}_prob_filtered.pfm".format(view))
        
        if os.path.exists(init_depth_map_path):
            depth_map, _ = read_pfm(init_depth_map_path)
            prob_map, _ = read_pfm(prob_map_path)
            depth_map[prob_map < prob_threshold] = 0
            save_pfm(out_depth_map_path, depth_map)


def imgcam_convert(image_folder, cam_folder, fusibile_workspace, nviews):
    # output dir
    fusion_cam_folger = os.path.join(fusibile_workspace, 'cams')
    fusion_image_folder = os.path.join(fusibile_workspace, 'images')
    os.makedirs(fusion_cam_folger,exist_ok=True)
    os.makedirs(fusion_image_folder,exist_ok=True)

    # cal proj cameras: [KR KT]
    for view in range(nviews):

        in_cam_file = os.path.join(cam_folder, "{:08d}_cam.txt".format(view))
        out_cam_file = os.path.join(fusion_cam_folger, "{:08d}.png.P".format(view))
        
        if os.path.exists(in_cam_file):
            cal_projection_matrix(in_cam_file, out_cam_file)

    # copy images to gipuma image folder
    for view in range(nviews):
        in_image_file = os.path.join(image_folder, "{:08d}.jpg".format(view))
        out_image_file = os.path.join(fusion_image_folder, "{:08d}.png".format(view))
        
        if os.path.exists(in_image_file):
            shutil.copy(in_image_file, out_image_file)

def to_gipuma(depthfiltered_folder, fusibile_workspace, nviews):
    gipuma_prefix = '2333__'
    for view in range(nviews):

        sub_depth_folder = os.path.join(fusibile_workspace, gipuma_prefix+"{:08d}".format(view))
        os.makedirs(sub_depth_folder, exist_ok=True)

        in_depth_pfm = os.path.join(depthfiltered_folder, "{:08d}_prob_filtered.pfm".format(view))
        out_depth_dmb = os.path.join(sub_depth_folder, 'disp.dmb')
        fake_normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')
        
        if os.path.exists(in_depth_pfm):
            image, _ = read_pfm(in_depth_pfm)
            write_gipuma_dmb(out_depth_dmb, image)
            fake_gipuma_normal(out_depth_dmb, fake_normal_dmb)

def depth_map_fusion(fusibile_workspace, fusibile_exe_path, disp_thresh, num_consistent):

    cam_folder = os.path.join(fusibile_workspace, 'cams')
    image_folder = os.path.join(fusibile_workspace, 'images')
    depth_min = 0.001
    depth_max = 100000
    normal_thresh = 360

    cmd = fusibile_exe_path
    cmd = cmd + ' -input_folder ' + fusibile_workspace + '/'
    cmd = cmd + ' -p_folder ' + cam_folder + '/'
    cmd = cmd + ' -images_folder ' + image_folder + '/'
    cmd = cmd + ' --depth_min=' + str(depth_min)
    cmd = cmd + ' --depth_max=' + str(depth_max)
    cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
    cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
    cmd = cmd + ' --num_consistent=' + str(num_consistent)
    print (cmd)
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dtu fusion parameter setting')
    parser.add_argument("-c", '--cut', action='store_true', help='cut img to keep the same size with eval')
    parser.add_argument("-f", '--filter', action='store_true', help='filter depth_map with prob_map')
    parser.add_argument("-m", '--move', action='store_true', help='move img and cal proj matrix')
    parser.add_argument("-g", '--gipuma', action='store_true', help='convert depth maps and fake normal maps')
    parser.add_argument("-d", '--depth_fusion', action='store_true', help='depth map fusion with gipuma')
    parser.add_argument('-r', '--remove', action='store_true', help='remove depth dir')


    args = parser.parse_args()
    print(args)

    eval_root = conf.eval_root
    scans = conf.scans
    img_folder = conf.img_folder
    camera_folder = conf.camera_folder

    # A.Crop the picture to match the network input
    if args.cut:
        logging.info("A.Crop the picture to match the network input")
        for scan in scans:
            nviews = conf.fusion_args[scan]["nviewss"]
            os.makedirs(os.path.join(eval_root, scan, conf.cut_img_folder), exist_ok=True)
            for vid in range(nviews):
                img_filename = os.path.join(eval_root, scan, img_folder, '{:0>8}.jpg'.format(vid))
                out_img_filename = os.path.join(eval_root, scan, conf.cut_img_folder, '{:0>8}.jpg'.format(vid))
                
                if os.path.exists(img_filename):
                    img = cv2.imread(img_filename)
                    h, w, _ = img.shape
                    h, w = cal_ncutpixs(h, w, conf.min_scale)
                    img = img[:h][:w]

                    logging.info("save location:" + out_img_filename+"("+str(w)+","+str(h)+")")
                    cv2.imwrite(out_img_filename, img)

    # B.Fusion
    logging.info("#>>>>>>>>>>>Start Fusion")
    for scan in scans:
        nviews = conf.fusion_args[scan]["nviewss"]
        check_view = conf.fusion_args[scan]["check_views"]
        prob_threshold = conf.fusion_args[scan]["prob_threshold"]
        disp_threshold = conf.fusion_args[scan]["disp_threshold"]

        logging.info("######current scan:"+scan+" nviews:"+str(nviews)+" check_view:"+str(check_view)
                     +" prob_threshold:"+str(prob_threshold)+" disp_threshold:"+str(disp_threshold))

        scan_folder = os.path.join(eval_root, scan)
        cam_folder = os.path.join(scan_folder, camera_folder)
        image_folder = os.path.join(scan_folder, conf.cut_img_folder)  # use crop img

        eval_dc_folder = os.path.join(conf.eval_folder, scan)
        depth_folder = os.path.join(eval_dc_folder, 'depth_est')
        prob_folder = os.path.join(eval_dc_folder, 'confidence')

        fusibile_workspace = os.path.join(conf.eval_folder, scan, "fuse")
        depthfiltered_folder = os.path.join(fusibile_workspace, "depth")
        os.makedirs(fusibile_workspace, exist_ok=True)
        os.makedirs(depthfiltered_folder, exist_ok=True)

        # probability filtering , save *_prob_filtered.pfm in depth_folder
        if args.filter:
            logging.info('>>1.filter depth map with probability map')
            probability_filter(depth_folder, prob_folder, nviews, prob_threshold, depthfiltered_folder)

        # data conversion
        if args.move:
            logging.info('>>2.move img and cal proj matrix')
            imgcam_convert(image_folder, cam_folder, fusibile_workspace, nviews)

        # convert depth maps and fake normal maps
        if args.gipuma:
            logging.info('>>3.convert depth maps and fake normal maps')
            to_gipuma(depthfiltered_folder, fusibile_workspace, nviews)

        # depth map fusion with gipuma
        if args.depth_fusion:
            logging.info('>>4.Run depth map fusion & filter')
            depth_map_fusion(fusibile_workspace, conf.fusibile_exe_path, disp_threshold, check_view)    # parser = argparse.ArgumentParser(description='Train parameter setting')

        # move ply file and remove depth map file
        if args.remove:
            logging.info('>>5.move ply file and delete scans outputs')
            os.makedirs(conf.col_path, exist_ok=True)

            # search ply file
            fuse_folder = os.path.join(conf.eval_folder, scan, "fuse")
            consis_folders = [f for f in os.listdir(fuse_folder) if f.startswith('consistencyCheck-')]
            consis_folders.sort()
            consis_folder = consis_folders[-1]
            source_ply = os.path.join(fuse_folder, consis_folder, 'final3d_model.ply')
            if conf.dataset == "dtu":
                scan_idx = int(scan[4:])
                target_ply = os.path.join(conf.col_path, 'ours{:03d}_l3.ply'.format(scan_idx))
            else:
                target_ply = os.path.join(conf.col_path, scan+'.ply')

            # move ply
            cmd = 'cp ' + source_ply + ' ' + target_ply
            logging.info(cmd)
            os.system(cmd)

            # To save disk space, del sence dir
            if os.path.exists(target_ply):
                cmd = "rm -r " + eval_dc_folder
                logging.info(cmd)
                os.system(cmd)
            else:
                print("remove error!")
           
