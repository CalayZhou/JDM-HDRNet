# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from skimage import io
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']


        img_bytes = self.file_client.get(filename)
        #for png
        # img = mmcv.imfrombytes(
        #     img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        #for tif
        img = io.imread(filename)
        # print(np.max(img))
        img = (img-np.min(img))/(np.max(img)-np.min(img))*255
        img = img.astype(np.uint8)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """



        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']

        # img = io.imread(filename)
        # img = img[:, :, np.newaxis]
        # img = img.transpose(2, 0 , 1)
        # img = img*255
        # #
        # img = img.astype(int)
        # img = img//32
        # results['gt_semantic_seg'] = img[0,:,:]#(906,1057,1)
        # return results


        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id

        # gt_semantic_seg = gt_semantic_seg[:,:,0]
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadReference(object):
    """Load reference image for semantic segmentation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self, file_client_args=dict(backend='disk'), imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # Replace this with the code to load your reference image
        if results.get('ref_prefix', None) is not None:
            filename = osp.join(results['ref_prefix'], results['ref_info']['seg_map'])
        else:
            filename = results['ref_info']['seg_map']
        # img_bytes = self.file_client.get(filename)
        # img_2 = mmcv.imfrombytes(
        #     img_bytes, flag='unchanged',
        #     backend=self.imdecode_backend).astype(np.uint8)
        img_2 = np.load(filename,mmap_mode=None,allow_pickle=False,fix_imports=True,encoding='ASCII')
        img_2 = img_2.transpose((1, 0, 2))[:, ::-1, :]
        img_2 = np.ascontiguousarray(img_2)
        # spatio_scale = 16
        # img_2_c = np.sum(img_2,axis=0)
        # img_2_c = np.sum(img_2_c,axis=0)
        '''
        img_2 = img_2.transpose(1, 0, 2)
        img_2 = img_2[:, ::-1, :].copy()
        img_2_h, img_2_w, _ = img_2.shape
        spatio_scale = 16
        img_2 = img_2[::(img_2_h//spatio_scale), ::(img_2_w//spatio_scale), :]
        # img_2 = img_2[:spatio_scale, :spatio_scale, :]
        img_2 = img_2[:spatio_scale, :spatio_scale, ::2]
        '''
        filename_sha = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        filename_nir = filename_sha.split('.')[0]+'_s.tif'
        img_sha = io.imread(filename_nir)
        img_sha = img_sha[:, :, np.newaxis]
        # img_sha = img_sha.transpose(2, 0 , 1)

        #
        input_max = np.max(img_2, axis = 2)
        nir_max = input_max / np.max(img_2)
        nir_max = nir_max[:, :, np.newaxis]
        nir = np.maximum(nir_max,img_sha)

        img_2 = img_2/nir

        # img_2_rescale = np.resize(img_2, (1, 1,10))
        # img_2_rescale_max = np.max(img_2_rescale)
        # img_2_rescale = img_2_rescale/img_2_rescale_max
        results['img_2'] = img_2

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadShadingImage(object):#calay 0906
    """Load reference image for semantic segmentation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self, file_client_args=dict(backend='disk'), imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.color_type = 'color'
        self.to_float32 = True

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # Replace this with the code to load your reference image
        # if results.get('ref_prefix', None) is not None:
        #     filename = osp.join(results['ref_prefix'], results['ref_info']['seg_map'])
        # else:
        #     filename = results['ref_info']['seg_map']
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        filename_nir = filename.split('.')[0]+'_s.tif'
        # img_bytes = self.file_client.get(filename_nir)
        # img = mmcv.imfrombytes(
        #     img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        img = io.imread(filename_nir)
        # if self.to_float32:
        #     img = img.astype(np.float32)/255.0
        # img = img[:, :, 1]
        img = img[:, :, np.newaxis]
        img = img.transpose(2, 0 , 1)
        img = img*255

        img = img.astype(int)
        img = img//32
        # nir_ori = (np.asarray(image * 255, dtype=np.int32) // 32 + 1) * 32
        results['gt_shading'] = img#(906,1057,1)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

