import os.path as osp
import pickle
import random
from mmcv.parallel import DataContainer as DC
import mmcv
import numpy as np
import torch
import scipy.io as scio
from ..builder import PIPELINES
from .formating import to_tensor
from .transforms import PhotoMetricDistortion


@PIPELINES.register_module()
class LoadHSIFromFile():

    @staticmethod
    def _read_mat(filename):
        data = scio.loadmat(filename)['data'].astype(np.float32)
        data = data.transpose(1, 0, 2)
        data = data[:, ::-1, :]
        data = data[:,:,36:164]
        # data = np.fromfile("%s" % filename, dtype=np.int32)
        # height = data[0]
        # width = data[1]
        # SR = data[2]
        # D = data[3]
        #
        # data = np.fromfile("%s" % filename, dtype=np.float32)
        # a = 7
        # average = data[a:a + SR]
        # a = a + SR
        # coeff = data[a:a + D * SR].reshape((D, SR))
        # a = a + D * SR
        # scoredata = data[a:a + height * width * D].reshape((height * width, D))
        #
        # temp = np.dot(scoredata, coeff)
        # print('dot')
        # data = (temp + average).reshape((height, width, SR))

        return data

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2') -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        hsiname = osp.join(results['img_prefix'], results['img_info']['hsi'])
        with open(hsiname, 'rb') as f:
            hsi = pickle.load(f)
        results['hsiname'] = hsiname
        results['hsi'] = hsi
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class ResizeHSI():

    def __init__(self, ratio=1, resize_hsi=False, min_size=None) -> None:
        self.ratio = ratio
        self.min_size = min_size
        self.resize_hsi = resize_hsi

    def _resize_img(self, results):
        img = results['img']
        img, w_scale = mmcv.imrescale(
            results['img'],
            self.ratio,
            interpolation='nearest',
            return_scale=True)
        scale_factor = np.array([w_scale, w_scale, w_scale, w_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = scale_factor

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            gt_seg = mmcv.imrescale(
                results[key], self.ratio, interpolation='nearest')
            results[key] = gt_seg

    def _resize_hsi(self, results):
        if self.resize_hsi:
            hsi = mmcv.imrescale(
                results['hsi'], self.ratio, interpolation='nearest')
            results['hsi'] = hsi

    def __call__(self, results):
        self._resize_img(results)
        self._resize_seg(results)
        if 'hsi' in results:
            self._resize_hsi(results)
        # self._resize_hsi(results)
        # h1, w1, _ = results['img'].shape
        # h2, w2, _ = results['hsi'].shape
        # assert h1 == h2 and w1 == w2, f'img has shape ({h1}, {w1}), hsi has shape ({h2}, {w2})'
        return results


@PIPELINES.register_module()
class PadHSI():

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_hsi(self, results):
        """Pad hsis according to ``self.size``."""
        padded_hsi = mmcv.impad(
            results['hsi'], shape=self.size, pad_val=self.pad_val)
        results['hsi'] = padded_hsi

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_hsi(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionHSI(PhotoMetricDistortion):

    def __init__(self, *args, **kwargs):
        super(PhotoMetricDistortionHSI, self).__init__(*args, **kwargs)

    def convert(self, img, alpha=1, beta=0):
        img = img * alpha + beta / 255.0
        return img

    def __call__(self, results):
        hsi = results['hsi']
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min())
        # random brightness
        hsi = self.brightness(hsi)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        hsi = self.contrast(hsi)

        results['hsi'] = hsi
        return results


@PIPELINES.register_module()
class DefaultFormatBundleHSI():

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        hsi = results['hsi']
        hsi = np.ascontiguousarray(hsi.transpose(2, 0, 1))
        results['hsi'] = DC(to_tensor(hsi), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ReplaceRGB():

    def __call__(self, result):
        result['img'] = result['hsi']
        del result['hsi']
        return result


@PIPELINES.register_module()
class HSIPCA():

    @staticmethod
    def _pca(hsi: np.ndarray, n_components=32):
        h, w, c = hsi.shape
        hsi = hsi - np.expand_dims(hsi.mean(axis=0), axis=0)
        hsi = hsi.reshape(h * w, c)
        u, s, v = np.linalg.svd(hsi, full_matrices=False)
        r = np.matmul(hsi, v.T[:, :n_components])
        return r.reshape(h, w, n_components)

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def __call__(self, result):
        hsi = result['hsi']
        hsi = self._pca(hsi, self.n_components)
        result['hsi'] = hsi
        return result
