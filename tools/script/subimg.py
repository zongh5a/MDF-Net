import cv2,sys
import numpy as np


def subimg(file1,file2):
    img1,img2=cv2.imread(file1),cv2.imread(file2)
    subimg=np.abs(img1-img2)
    print(subimg.shape,subimg.min(),subimg.max())

    cv2.imshow("subimg",subimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    subimg(sys.argv[1],sys.argv[2])
