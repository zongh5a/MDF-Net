import re,cv2,sys
import numpy as np
import matplotlib.pyplot as plt

#file header
""""
Pf
400 296
-1.000000
"""

def catPfm(path):
    file=open(path, 'rb')
    header = file.readline().decode('UTF-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    depth_image = cv2.flip(data, 0)

    ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
    print('value range: ', ma.min(), ma.max())

    plt.title(path)
    plt.imshow(depth_image, 'rainbow')
    plt.show()

if __name__=="__main__":
    catPfm(sys.argv[1])
