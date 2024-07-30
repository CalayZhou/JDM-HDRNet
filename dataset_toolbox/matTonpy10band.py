import os
import scipy.io as scio
import cv2
import numpy as np

self_data_path = "/home/calay/DATASET/mat_final/"
self_save_path = "/home/calay/DATASET/mat_final/spec_npy10band/"
if not os.path.exists(self_save_path):
    os.makedirs(self_save_path)

file_list = os.listdir(self_data_path)
for file_i in file_list:
    if not os.path.exists(os.path.join(self_data_path, file_i)):
        continue
    print(file_i)
    spec = scio.loadmat(os.path.join(self_data_path, file_i))['data'].astype(np.float32)
    #print(np.max(spec))
    spec_10band = np.zeros((1057,960,10))
    # spec_10band = np.zeros((np.shape(spec)[0],np.shape(spec)[1],10))
    spec_10band = np.asarray(spec_10band, dtype=np.float32)
    for i in range(10):
        spec_10band[:,:,i] = np.mean(spec[:,:,i*17: i*17+17],axis=2)
    print(np.max(spec_10band))
    save_path = os.path.join(self_save_path, file_i.split('.')[0]+'.npy')
    # np.save(save_path,spec_10band,allow_pickle=False,fix_imports=True)
    np.save(save_path,spec_10band)

