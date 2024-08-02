# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from torchvision.transforms.functional import resize as resize_tensor
from PIL import Image

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 backbone_2,# wyb1219
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        # if pretrained is not None:
        #     assert backbone_2.get('pretrained') is None, \
        #         'both backbone and segmentor set pretrained weight'
        #     backbone_2.pretrained = pretrained

        self.backbone_2 = builder.build_backbone(backbone_2)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, **kwargs):
        """Extract features from images."""
        # print('img',img.shape)
        try:
            x = self.backbone(img, **kwargs)
            if self.with_neck:
                x = self.neck(x, **kwargs)
        except TypeError:
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
        return x
    def extract_feat_2(self, img_2, **kwargs):
        """Extract features from images."""
        try:
            x = self.backbone_2(img_2, **kwargs)
            if self.with_neck:
                x = self.neck(x, **kwargs)
        except TypeError:
            x = self.backbone_2(img_2)
            if self.with_neck:
                x = self.neck(x)
        return x

    def encode_decode(self, img, img_2, img_metas, **kwargs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        """
        img_2 = img_2.to(torch.float32)/65535.0
        spatio_scale = 16
        img_2= resize_tensor(img_2, (spatio_scale, spatio_scale), Image.BILINEAR)
        #img_2_np = img_2.cpu().numpy()[0, 0, :, :][:, :, np.newaxis]  # .transpose(1, 2, 0)
        #img_np = img.cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 255
        # gt_shading_np = gt_shading.cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 32
        # cv2.imwrite('img_2_np_test.jpg', img_2_np * 255)
        # cv2.imwrite('img_np_test.jpg', img_np)

        if img.shape[2:] != img_2.shape[2:]:
            img_2 = F.interpolate(img_2, size=img.shape[2:], mode='bilinear', align_corners=True)


        x = self.extract_feat(img, **kwargs)
        x_2 = self.extract_feat_2(img_2, **kwargs)  # 提取第二路输入的特征
        x = list(x)
        x_2 = list(x_2)
        layer_i = -1
        x[layer_i] = torch.cat([x[layer_i], x_2[layer_i]], dim=1)
        x = tuple(x)

        if self.with_auxiliary_head:
            out_nir = self._auxiliary_head_forward_test(x, img_metas)#calay 1008
            out_nir = resize(
                input=out_nir,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # return out_nir#
        out_concat = torch.cat([out,out_nir],dim=1)
        return out_concat

    def _decode_head_forward_train(
        self,
        x,
        img_metas,
        gt_semantic_seg,
    ):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            x,
            img_metas,
            gt_semantic_seg,
            self.train_cfg,
        )

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for auxiliary head in
        inference."""
        seg_logits = self.auxiliary_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(
        self,
        x,
        img_metas,
        gt_semantic_seg,
    ):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    # def forward_dummy(self, img, **kwargs):
    #     """Dummy forward function."""
    #     seg_logit = self.encode_decode(img, None)
    #
    #     return seg_logit

    def forward_dummy(self, img, img_2, **kwargs):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, img_2, None)

        return seg_logit

    def forward_train(self, img, img_2, img_metas, gt_shading, gt_semantic_seg, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        img_2 = img_2.to(torch.float32)/65535.0
        img_2 = img_2.permute(0, 3, 1, 2).contiguous()
        spatio_scale = 16
        img_2= resize_tensor(img_2, (spatio_scale, spatio_scale), Image.BILINEAR)
        '''
        img_2_np = img_2.cpu().numpy()[0, 0, :, :][:, :, np.newaxis]  # .transpose(1, 2, 0)
        img_np = img.cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 255
        gt_shading_np = gt_shading.cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 32
        gt_semantic_seg_np = gt_semantic_seg.cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 32
        cv2.imwrite('img_2_np.jpg', img_2_np * 255)
        cv2.imwrite('img_np.jpg', img_np)
        cv2.imwrite('gt_shading_np.jpg', gt_shading_np)
        cv2.imwrite('gt_semantic_seg_np.jpg', gt_semantic_seg_np)
        '''
        if img.shape[2:] != img_2.shape[2:]:
            img_2 = F.interpolate(img_2, size=img.shape[2:], mode='bilinear', align_corners=True)
        x = self.extract_feat(img, **kwargs)

        x = list(x)
        x_2 = self.extract_feat_2(img_2, **kwargs)
        x_2 = list(x_2)
        layer_i = -1
        x[layer_i] = torch.cat([x[layer_i], x_2[layer_i]], dim=1)
        x = tuple(x)
        losses = dict()

        loss_decode = self._decode_head_forward_train(
            x,
            img_metas,
            gt_semantic_seg,
        )
        losses.update(loss_decode)

        gt_shading = gt_shading.to(torch.int64)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_shading)#calay 0906
            losses.update(loss_aux)
        return losses

    # TODO refactor
    def slide_inference(self, img,img_2, img_meta, rescale, **kwargs):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_img_2 = img_2[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img,crop_img_2, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img,img_2, img_meta, rescale, **kwargs):
        """Inference with full image."""
        seg_logit = self.encode_decode(img,img_2, img_meta, **kwargs)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            if 'img_shape' in img_meta[0]:
                imgx, imgy = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :imgx, :imgy]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img,img_2, img_meta, rescale, **kwargs):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img,img_2, img_meta, rescale, **kwargs)
        else:
            seg_logit = self.whole_inference(img,img_2, img_meta, rescale, **kwargs)
        seg_logit_seg = seg_logit[:,:5,:,:]
        seg_logit_nir = seg_logit[:,5:13,:,:]
        output_seg = F.softmax(seg_logit_seg, dim=1)
        output_nir = F.softmax(seg_logit_nir, dim=1)
        output = torch.cat([output_seg,output_nir],dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img,img_2, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        if 'hsi' in kwargs:
            kwargs['hsi'] = kwargs['hsi'][0]
        seg_logit = self.inference(img,img_2, img_meta, rescale, **kwargs)
        seg_logit_seg = seg_logit[:,:5,:,:]
        seg_logit_nir = seg_logit[:,5:13,:,:]
        sseg_pred_seg = seg_logit_seg.argmax(dim=1)
        seg_pred_nir = seg_logit_nir.argmax(dim=1)
        seg_pred = torch.cat([sseg_pred_seg,seg_pred_nir],dim=0)
        # seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs,imgs_2, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0],imgs_2[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i],imgs_2[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
