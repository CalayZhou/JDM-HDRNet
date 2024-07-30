

import os
import numpy as np
import cv2
import scipy.io as scio

data_path = "/home/calay/DATASET/mat_final/"
output_path = "/home/calay/DATASET/nir/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

file_list = os.listdir(data_path)

for file_i in file_list:
    if not os.path.exists(os.path.join(data_path, file_i)):
        continue
    print(file_i)
    spec = scio.loadmat(os.path.join(data_path, file_i))['data'].astype(np.float32)
    spec = spec[:, :, 137:176]#850m——1000nm
    spec = np.sum(spec,axis=2)
    spec = spec/np.max(spec)
    spec = cv2.rotate(spec,rotateCode = cv2.ROTATE_90_CLOCKWISE)
    # cv2.imwrite(output_path +file_i.split('.')[0] + '.png',spec*255)
    cv2.imwrite(output_path +file_i.split('.')[0] + '.tif',spec)
    cv2.imwrite(output_path +file_i.split('.')[0] + '.png',spec*255)
