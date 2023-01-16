import numpy as np
import os

# read for train and eval
def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsic = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsic = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    return intrinsic, extrinsic

def save_paras(intrinsic, extrinsic,save_location, other="425 935"):
    content = "extrinsic" + "\n"
    for r in extrinsic:
        for i in r:
            content += str(i) + " "
        content += "\n"
    content += "\n" + "intrinsic" + "\n"
    for r in intrinsic:
        for i in r:
            content += str(i) + " "
        content += "\n"
    content += "\n" + other

    with open(save_location, "w") as f:
        f.write(content)

def scaleinmatrix(in_path,out_path):
    for i in range(49):
        cam ='{:08d}_cam.txt'.format(i)
        with open(in_path+cam, "r") as fin:
            intrinsic, extrinsic = read_cam_file(in_path+cam)

        intrinsic[:2,] = intrinsic[:2,]*4

        save_paras(intrinsic, extrinsic, out_path+cam)

if __name__=="__main__":
    in_path = 'F:\datasets\mvs_datasets\\train\dtu640x512\Cameras\\train\\'
    out_path = r"F:\datasets\mvs_datasets\\train\dtu640x512\Cameras\640x512\\"
    os.makedirs(out_path,exist_ok=True)

    scaleinmatrix(in_path, out_path)
    print("...ok")
