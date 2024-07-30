import numpy as np
import os
import torch
from PIL import Image
from skimage import io
from torchvision import transforms
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset
from utils import get_files

class BaseDataset(Dataset):
    def get_tif(self, path, is_jdm_predict):
        memory_tif = []
        memory_tif_input = {}
        memory_tif_output = {}
        memory_tif_nir = {}
        memory_spec = {}
        for file_name in os.listdir(path+'/target'):
            fname = file_name
            input = io.imread(os.path.join(path, 'source', fname.split('.')[0] + '.tif'))
            output = io.imread(os.path.join(path, 'target', fname.split('.')[0] + '.tif'))
            if is_jdm_predict:
                nir = io.imread(os.path.join(path, 'nir_jdm', fname.split('.')[0] + '.png'))
                nir = nir[:, :, 0]
                nir = (nir // 32 + 1)#1-8
                nir = nir / 8.0#0.125-1.0
            else:
                nir = io.imread(os.path.join(path, 'nir', fname.split('.')[0] + '.tif'))
            spec = np.load(os.path.join(path, 'spec_npy10band', fname.split('.')[0] + '.npy'), mmap_mode=None,
                           allow_pickle=False, fix_imports=True, encoding='ASCII')

            memory_tif_input.update({fname.split('.')[0]:input})
            memory_tif_output.update({fname.split('.')[0]:output})
            memory_tif_nir.update({fname.split('.')[0]:nir})
            memory_spec.update({fname.split('.')[0]:spec})
        memory_tif.append(memory_tif_input)
        memory_tif.append(memory_tif_output)
        memory_tif.append(memory_tif_nir)
        memory_tif.append(memory_spec)
        return memory_tif

    def load_img_hdr(self, fname,read_memory = False):
        if read_memory:
            input = self.memory_tif[0][fname.split('.')[0]]
            output = self.memory_tif[1][fname.split('.')[0]]
            nir_ori = self.memory_tif[2][fname.split('.')[0]]
            spec = self.memory_tif[3][fname.split('.')[0]]
        else:
            input = io.imread(os.path.join(self.data_path, 'source', fname.split('.')[0] + '.tif'))
            output = io.imread(os.path.join(self.data_path, 'target', fname.split('.')[0] + '.tif'))
            if self.params['jdm_predict']:
                nir_ori = io.imread(os.path.join(self.data_path, 'nir_jdm', fname.split('.')[0] + '.png'))
                nir_ori = nir_ori[:, :, 0]
                nir_ori = (nir_ori // 32 + 1)#1-8
                nir_ori = nir_ori / 8.0#0.125-1.0
            else:
                nir_ori = io.imread(os.path.join(self.data_path, 'nir', fname.split('.')[0] + '.tif'))
            spec = np.load(os.path.join(self.data_path, 'spec_npy10band', fname.split('.')[0] + '.npy'), mmap_mode=None,
                           allow_pickle=False, fix_imports=True, encoding='ASCII')
        if self.params['jdm_predict']:
            seg = io.imread(os.path.join(self.data_path, 'seg_jdm', fname.split('.')[0] + '.png'))
        else:
            seg = io.imread(os.path.join(self.data_path, 'seg', fname.split('.')[0] + '.png'))

        spec = spec.transpose((1, 0, 2))[:,::-1,:]
        spec = np.ascontiguousarray(spec)

        #############segmentation#################
        #CLASSES = ('sky','tree','building','trunk','road')
        #PALETTE =[[19,19,194], [43,139,3], [248,232,109], [78,50,12],[102,102,100]]
        sky_mask = np.where((seg[:,:,0]==19)&(seg[:,:,1]==19)&(seg[:,:,2]==194),1,0)
        tree_mask = np.where((seg[:,:,0]==43)&(seg[:,:,1]==139)&(seg[:,:,2]==3),1,0)
        building_mask = np.where((seg[:,:,0]==248)&(seg[:,:,1]==232)&(seg[:,:,2]==109),1,0)
        trunk_mask = np.where((seg[:,:,0]==78)&(seg[:,:,1]==50)&(seg[:,:,2]==12),1,0)
        road_mask = np.where((seg[:,:,0]==102)&(seg[:,:,1]==102)&(seg[:,:,2]==100),1,0)
        others_mask = np.where((seg[:,:,0]==8)&(seg[:,:,1]==8)&(seg[:,:,2]==8),1,0)

        sky_mask = np.expand_dims(sky_mask,axis=2)
        tree_mask = np.expand_dims(tree_mask,axis=2)
        building_mask = np.expand_dims(building_mask,axis=2)
        trunk_mask = np.expand_dims(trunk_mask,axis=2)
        road_mask = np.expand_dims(road_mask,axis=2)
        others_mask = np.expand_dims(others_mask,axis=2)

        material_mask = np.concatenate((sky_mask,tree_mask,building_mask,trunk_mask,\
                                         road_mask,others_mask ),axis=2)

        spec = np.asarray(spec, dtype=np.float32)
        input = np.asarray(input, dtype=np.float32)
        output = np.asarray(output, dtype=np.float32)
        material_mask = np.asarray(material_mask, dtype=np.float32)
        nir_ori = np.asarray(nir_ori, dtype=np.float32)

        #constraint the nir range
        nir_ori = nir_ori[:,:,np.newaxis]
        input_max = np.max(input,axis = 2)
        nir_max = input_max / np.max(input)
        nir_max = nir_max[:, :, np.newaxis]
        nir = np.maximum(nir_max,nir_ori)

        input = torch.from_numpy(input.transpose((2, 0, 1)))
        nir = torch.from_numpy(nir.transpose((2, 0, 1)))
        output = torch.from_numpy(output.transpose((2, 0, 1)))
        spec = torch.from_numpy(spec.transpose((2, 0, 1)))
        material_mask = torch.from_numpy(material_mask.transpose((2, 0, 1)))

        return input, output, spec, material_mask,nir

    def __len__(self):
        return len(self.input_paths)


class Train_Dataset(BaseDataset):
    """Class for training images."""

    def __init__(self, params=None):
        self.data_path = params['train_data_dir']
        self.input_paths = get_files(os.path.join(self.data_path, 'source'))
        self.memory_tif = self.get_tif(self.data_path,params['jdm_predict'])
        self.input_res = params['input_res']
        self.output_res = params['output_res']

        self.augment = transforms.Compose([
            transforms.RandomCrop(self.output_res),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        self.params = params

    def __getitem__(self, idx):

        fname = self.input_paths[idx].split('/')[-1]
        if self.params['hdr']:
            input, output, spec, material_mask,nir = self.load_img_hdr(fname,read_memory=True)
        # Check dimensions before crop
        assert input.shape == output.shape
        assert self.output_res[0] <= input.shape[2]
        assert self.output_res[1] <= input.shape[1]
        # Crop
        inout = torch.cat([input,output,spec,material_mask,nir],dim=0)
        inout = self.augment(inout)

        full = inout[:3,:,:]
        low = resize(full, (self.input_res, self.input_res), Image.BILINEAR)
        output = inout[3:6,:,:]

        spec = inout[6:6+spec.shape[0],:,:]
        material_mask = inout[6+spec.shape[0]:6+spec.shape[0]+6,:,:]
        nir = inout[6+spec.shape[0]+6:6+spec.shape[0]+7,:,:]
        spec_tmp = resize(spec, (self.params['spec_size'], self.params['spec_size']), Image.BILINEAR)
        spec_low = resize(spec_tmp, (self.input_res, self.input_res), Image.BILINEAR)

        return low, full, output, spec_low,material_mask,nir

class Eval_Dataset(BaseDataset):
    """Class for validation images."""

    def __init__(self, params=None):
        self.data_path = params['eval_data_dir']
        self.input_paths = get_files(os.path.join(self.data_path,  'source'))#'input'))
        # self.memory_tif = self.get_tif(self.data_path)
        self.input_res = params['input_res']
        self.params = params

    def __getitem__(self, idx):
        fname = self.input_paths[idx].split('/')[-1]
        if self.params['hdr']:
            full, output, spec, material_mask,nir = self.load_img_hdr(fname,read_memory=False)
        low = resize(full, (self.input_res, self.input_res), Image.BILINEAR)
        spec_tmp = resize(spec, (self.params['spec_size'], self.params['spec_size']), Image.BILINEAR)
        spec_low = resize(spec_tmp, (self.input_res, self.input_res), Image.BILINEAR)
        return low, full, output, spec_low, material_mask, nir
