import os

"""
 logging info format 
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s: %(message)s")


"""
eval ouput depth and confidence location
"""
eval_folder = "/hy-tmp/outputs"    #"../../outputs"   # #eval output location


""" 
dataset args
"""
root_dir = os.path.join("/hy-nas")
# root_dir = os.path.join("/home", "user22", "yu", "mvs_datasets")

dataset = "dtu"
#dataset = "tanks_intermediate"
#dataset = "tanks_advanced"
#>>>>> dtu
if dataset == "dtu":
    eval_root = os.path.join(root_dir, "dtu1600x1200")
    # dtu_labels = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
    dtu_labels = [11,]#1, 4, 9, 10, 11, 12, 13, 15, 23, 24, ]
    # dtu_labels = [29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
    scans = ["scan"+str(label) for label in dtu_labels]
    img_folder = "images"
    camera_folder = "cams"
    fusion_args = {
        scan: {"nviewss": 49, "prob_threshold": 0.6, "check_views": 3, "disp_threshold": 0.25} for scan in scans    #0.6
    }
elif dataset == "tanks_intermediate":
    eval_root = os.path.join(root_dir, "TankandTemples", "intermediate")
    #scans = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
    scans = ['Family', 'Panther', 'Playground', 'Train']
    img_folder = "images"
    camera_folder = "cams_1"
    fusion_args = {
    'Family': {"nviewss": 152, "prob_threshold": 0.8, "check_views": 4, "disp_threshold": 0.25},     # 151+1  -0.7 5 +200 ucs
    'Francis': {"nviewss": 302, "prob_threshold": 0.6, "check_views": 7, "disp_threshold": 0.2},    # 301+1  -10 7 +300 ucs
    'Horse': {"nviewss": 151, "prob_threshold": 0.6, "check_views": 4, "disp_threshold": 0.25},      # 150+1  -7 4 -20
    'Lighthouse': {"nviewss": 309, "prob_threshold": 0.6, "check_views": 5, "disp_threshold": 0.3}, # 308+1  +4 1 0.01 6    +500 ucs
    'M60': {"nviewss": 313, "prob_threshold": 0.6, "check_views": 4, "disp_threshold": 0.2},       # 312+1  -6 0.6 0.006 4    +30
    'Panther': {"nviewss": 314, "prob_threshold": 0.8, "check_views": 4, "disp_threshold": 0.2},    # 313+1  -2 1   0.005 6    +70
    'Playground': {"nviewss": 307, "prob_threshold": 0.8, "check_views": 5, "disp_threshold": 0.25},# 306+1 +7 1 0.005 6    +10
    'Train': {"nviewss": 301, "prob_threshold": 0.8, "check_views": 5, "disp_threshold": 0.25},     # 300+1 +1 1 0.005 5    -10
    }
elif dataset == "tanks_advanced":
    eval_root = os.path.join(root_dir, "TankandTemples", "advanced")
    # scans = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
    scans = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Temple']
    img_folder = "images"
    camera_folder = "cams_1"
    fusion_args = {
        #'Auditorium': {"nviewss": 302, "prob_threshold": 0.8, "check_views": 5, "disp_threshold": 0.25}, # 301+1->298    3
        'Auditorium': {"nviewss": 302, "prob_threshold": 0.8, "check_views": 3, "disp_threshold": 0.25}, # 301+1->298    3
        'Ballroom': {"nviewss": 324, "prob_threshold": 0.8, "check_views": 5, "disp_threshold": 0.25},   # 323+1->323 - 69-th ** 4
        'Courtroom': {"nviewss": 301, "prob_threshold": 0.8, "check_views": 5, "disp_threshold": 0.25},  # 300+1 4
        'Museum': {"nviewss": 301, "prob_threshold": 0.8, "check_views": 5, "disp_threshold": 0.25},     # 300+1 4
        'Palace': {"nviewss": 509, "prob_threshold": 0.8, "check_views": 5, "disp_threshold": 0.25}, # 508+1->504 ** 5
        #'Temple': {"nviewss": 302, "prob_threshold": 0.8, "check_views": 5, "disp_threshold": 0.25}, # 301+1 4
        'Temple': {"nviewss": 302, "prob_threshold": 0.8, "check_views": 4, "disp_threshold": 0.15}, # 301+1 4
    }
else:
    print("error dataset!")


"""
cut img dir
"""
min_scale = 5   # the min scale of image, use it to cut img
cut_img_folder = "images_cut"


"""
fusion parameters
"""
# fusibile_exe_path = "/data/user10/fusibile/fusibile"
fusibile_exe_path = "/root/fusibile/fusibile"

"""
collect path
"""
col_path = "/hy-tmp/oursply"#"/data/user10/SampleSet/MVSData/Points/ours"

