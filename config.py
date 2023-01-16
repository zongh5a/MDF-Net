import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # ,3,4,5,6,7'
print("all cuda:", os.environ['CUDA_VISIBLE_DEVICES'])


import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s: %(message)s")

import random,numpy,torch
seed_id = 1
random.seed(seed_id)
numpy.random.seed(seed_id)


""" 
    basic class: args
"""
import torch
torch.manual_seed(seed_id)


class Args():
    def show_args(self):
        print(self.__class__.__name__+":")
        for k, v in self.__dict__.items():
            print("\t"+k, ":", v)

    def get_device(self, parallel):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.deterministic = False
            if parallel:
                torch.cuda.manual_seed_all(seed_id)
                # os.environ["NCCL_DEBUG"] = "INFO"
                return os.environ['CUDA_VISIBLE_DEVICES']
            else:
                torch.cuda.manual_seed(seed_id)
                return torch.device("cuda:0")
        else:
            return torch.device("cpu")

""" 
    trian args: dtu
"""
class TrainArgs(Args):
    def __init__(self):
        self.nviews = 5
        self.robust = True

        self.start_epoch = 1
        self.max_epoch = 30
        self.batch_size = 4#2
        self.nworks = 2
        self.lr = 1e-3
        self.factor = 0.9

        # self.val_nviews = 5
        self.pth_path='pth'  # pth file save path
        os.makedirs(self.pth_path, exist_ok=True)

        self.parallel = True
        self.DEVICE = self.get_device(self.parallel)

        self.show_args()


""" 
    train args: blendedmvs
"""
class BlendedMVSArgs(Args):
    def __init__(self):
        self.nviews = 5 #7
        self.robust = True

        self.start_epoch = 1
        self.max_epoch = 30
        self.batch_size = 6
        self.nworks = 3
        self.lr = 1e-3
        self.factor = 0.9

        self.pth_path='pth'  # pth file save path
        os.makedirs(self.pth_path, exist_ok=True)
        self.parallel = True
        self.DEVICE = self.get_device(self.parallel)

        self.show_args()


"""
    eval args: dtu, tanks
"""
class EvalArgs(Args):
    def __init__(self):
        self.output_path = "/hy-tmp/outputs"
        os.makedirs(self.output_path, exist_ok=True)

        self.parallel = False
        self.DEVICE = self.get_device(self.parallel)


class EvalDTU(EvalArgs):
    def __init__(self):
        super(EvalDTU, self).__init__()
        self.batch_size = 1
        self.nworks = 1
        self.nviews = 5

        self.show_args()


class EvalTanks(EvalArgs):
    def __init__(self):
        super(EvalTanks, self).__init__()
        self.batch_size = 1
        self.nworks = 1
        self.nviews = 11

        self.show_args()


""" 
    dataset args 
"""
class DatasetsArgs(Args):
    def __init__(self):
        self.root_dir = os.path.join("/hy-tmp")

class LoadDTU(DatasetsArgs):
    def __init__(self):
        super(LoadDTU, self).__init__()
        self.train_root= os.path.join(self.root_dir, "dtu640x512")  #dtu640x512 dtu160x128
        self.train_pair = os.path.join(self.train_root,"Cameras","pair.txt")
        self.train_label = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                      45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                      74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                      121, 122, 123, 124, 125, 126, 127, 128]
        self.train_lighting_label = [0, 1, 2, 3, 4, 5, 6]
        self.train_robust = True

        # self.val_label = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
        # self.val_lighting_label = [3]

        self.eval_root = os.path.join(self.root_dir, "dtu1600x1200")
        self.eval_pair = os.path.join(self.eval_root,"pair.txt")
        # self.eval_label = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
        self.eval_label = [11,]

        self.show_args()


"""
    BlendedMVS dataset
"""
class LoadBlendedMVS(DatasetsArgs):
    def __init__(self):
        super(LoadBlendedMVS, self).__init__()
        self.train_root = os.path.join(self.root_dir, "blendedmvs768x576")

        self.show_args()


""" 
    tanks dataset args 
"""
class LoadTanks(DatasetsArgs):
    def __init__(self, tanks_set = "intermediate"):   #"advanced"
        super(LoadTanks, self).__init__()
        self.eval_root = os.path.join(self.root_dir, "TankandTemples", tanks_set)
        if tanks_set == "intermediate":
            self.scenelist = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
            # self.scenelist = ['Family',]
        elif tanks_set == "advanced":
            self.scenelist = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Temple', 'Palace']
            #self.scenelist = ['Palace']

        self.show_args()


""" 
    net args
"""
import torch.nn as nn
from net import core
from net.unit import scale, backbone, regress, refine
from net.unit.depthhypos import HyposByFit
from net.unit.homoaggregate import VectorAggregate
from net.unit.regular import RegularNet_4Scales, RegularNet_3Scales
stages = 4
# scale matrix method
scale = scale.scale_cam
# Feature map extraction network
chs = (8, 16, 32, 64)
Backbone= backbone.FPN_4Scales(chs)
#Depth hypothesis method
ndepths = (48, 24, 8) # Number of depth assumption planes
curve_calss = [None, "gauss1", "laplace"]
prob_thresh = (0.0, 0.95, 1e-5)
Depth_hypos = nn.ModuleList([HyposByFit(ndepths[i], curve_calss[i], prob_thresh[i])
                             for i in range(stages-1)])
# Cost volume construction and aggregation method
ngroups = (32, 16, 8)
Homoaggre = nn.ModuleList([VectorAggregate(ngroups[i])
                           for i in range(stages-1)])
# 3D convolution regularization method
Regular0 = RegularNet_3Scales(ngroups[0])
Regular = nn.ModuleList([RegularNet_4Scales(in_ch)
                         for in_ch in ngroups[1:]])
Regular.insert(0, Regular0)
# Depth and confidence regression method
Regress = [regress.depth_regression, regress.confidence_regress]
# refine
Refine = refine.RefineNet2()    #RefineNet()
# model
model = core.CoreNet(Backbone, Depth_hypos, scale, Homoaggre, Regular, Regress, Refine)
                     
                     
if __name__=="__main__":
    train_args = TrainArgs()
    finetune_args = BlendedMVSArgs()
    evaldtu_args = EvalDTU()
    evaltanks_args = EvalTanks()
    blendedmvs_args = LoadBlendedMVS()
    dtu_args = LoadDTU()
    tanks_args = LoadTanks()

    exit()
   