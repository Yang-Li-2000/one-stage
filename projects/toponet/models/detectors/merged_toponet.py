# ---------------------------------------------------------------------------------------#
# Graph-based Topology Reasoning for Driving Scenes (https://arxiv.org/abs/2304.05277)  #
# Source code: https://github.com/OpenDriveLab/TopoNet                                  #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
# ---------------------------------------------------------------------------------------#

import time
import copy
import numpy as np
import torch

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from ...utils.builder import build_bev_constructor
from mmdet.models.utils.transformer import inverse_sigmoid


@DETECTORS.register_module()
class MergedTopoNet(MVXTwoStageDetector):

    def __init__(self,
                 bev_constructor=None,
                 bbox_head=None,
                 lane_head=None,
                 video_test_mode=False,
                 **kwargs):

        super(MergedTopoNet, self).__init__(**kwargs)

        if bev_constructor is not None:
            self.bev_constructor = build_bev_constructor(bev_constructor)

        if bbox_head is not None:
            bbox_head.update(train_cfg=self.train_cfg.bbox)
            self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None

        if lane_head is not None:
            lane_head.update(train_cfg=self.train_cfg.lane)
            self.pts_bbox_head = build_head(lane_head)
        else:
            self.pts_bbox_head = None

        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C,
                                  H, W))
            else:
                img_feats_reshaped.append(
                    img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue,
                                               len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.bev_constructor(img_feats, img_metas, prev_bev)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_lanes_3d=None,
                      gt_lane_labels_3d=None,
                      gt_lane_adj=None,
                      gt_lane_lcte_adj=None,
                      gt_bboxes_ignore=None,
                      ):

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        else:
            prev_bev = None

        img_metas = [each[len_queue - 1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        front_view_img_feats = [lvl[:, 0] for lvl in img_feats]
        batch_input_shape = tuple(img[0, 0].size()[-2:])
        bbox_img_metas = []
        for img_meta in img_metas:
            bbox_img_metas.append(
                dict(
                    batch_input_shape=batch_input_shape,
                    img_shape=img_meta['img_shape'][0],
                    scale_factor=img_meta['scale_factor'][0],
                    crop_shape=img_meta['crop_shape'][0]))
            img_meta['batch_input_shape'] = batch_input_shape

        # # dense_heads.deformable_detr_head.CustomDeformableDETRHead
        # bbox_outs = self.bbox_head(front_view_img_feats, bbox_img_metas)
        # # dense_heads.toponet_head.TopoNetHead: (gnn in ffn uses te_feature)
        # outs = self.pts_bbox_head(img_feats, bev_feats, img_metas, None, None)

        # 1. prepare inputs
        te_feats = None
        te_cls_scores = None
        bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)

        # 2. go through the first half
        outputs_te_head_first_half = self.bbox_head.forward_first_half(front_view_img_feats, bbox_img_metas)
        outputs_te_transformer_first_half = self.bbox_head.transformer.forward_first_half(*outputs_te_head_first_half)
        outputs_cl_head_first_half = self.pts_bbox_head.forward_first_half(img_feats, bev_feats, img_metas, te_feats, te_cls_scores)
        outputs_cl_transformer_first_half = self.pts_bbox_head.transformer.forward_first_half(
            outputs_cl_head_first_half['mlvl_feats'],
            outputs_cl_head_first_half['bev_feats'],
            outputs_cl_head_first_half['object_query_embeds'],
            outputs_cl_head_first_half['bev_h'],
            outputs_cl_head_first_half['bev_w'],
            img_metas=outputs_cl_head_first_half['img_metas']
        )

        # 3. manually go through decoders
        num_decoder_layers = len(self.bbox_head.transformer.decoder.layers)
        if len(self.bbox_head.transformer.decoder.layers) != len(self.pts_bbox_head.transformer.decoder.layers):
            raise NotImplementedError('not implemented when two decoders have different number of layers')

        # for te decoder
        query_te = outputs_te_transformer_first_half['query']
        memory_te = outputs_te_transformer_first_half['memory']
        query_pos_te = outputs_te_transformer_first_half['query_pos']
        init_reference_out_te = outputs_te_transformer_first_half['init_reference_out']
        mask_flatten_te = outputs_te_transformer_first_half['mask_flatten']
        reference_points_te = outputs_te_transformer_first_half['reference_points']
        spatial_shapes_te = outputs_te_transformer_first_half['spatial_shapes']
        level_start_index_te = outputs_te_transformer_first_half['level_start_index']
        valid_ratios_te = outputs_te_transformer_first_half['valid_ratios']
        reg_branches_te = outputs_te_transformer_first_half['reg_branches']
        kwargs_te = outputs_te_transformer_first_half['kwargs']

        # for cl decoder
        query_cl = outputs_cl_transformer_first_half['query']
        key_cl = outputs_cl_transformer_first_half['key']
        value_cl = outputs_cl_transformer_first_half['value']
        query_pos_cl = outputs_cl_transformer_first_half['query_pos']
        reference_points_cl = outputs_cl_transformer_first_half['reference_points']
        spatial_shapes_cl = outputs_cl_transformer_first_half['spatial_shapes']
        level_start_index_cl = outputs_cl_transformer_first_half['level_start_index']
        init_reference_out_cl = outputs_cl_transformer_first_half['init_reference_out']
        kwargs_cl = outputs_cl_transformer_first_half['kwargs']

        # for te decoder layer
        intermediate_te = []
        intermediate_reference_points_te = []

        # for cl decoder layer
        intermediate_cl = []
        intermediate_reference_points_cl = []

        # looping over each decoder layer
        for lid in range(num_decoder_layers):

            # 0. input preparation (te)
            if reference_points_te.shape[-1] == 4:
                reference_points_input_te = reference_points_te[:, :, None] * torch.cat([valid_ratios_te, valid_ratios_te], -1)[:, None]
            else:
                assert reference_points_te.shape[-1] == 2
                reference_points_input_te = reference_points_te[:, :, None] * valid_ratios_te[:, None]

            # 0. input preparation (cl)
            reference_points_input_cl = reference_points_cl[..., :2].unsqueeze(2)

            # 1. self-attention after concatenating queries
            query_te, query_cl, decoder_self_attention_q_te, decoder_self_attention_q_cl, decoder_self_attention_k_te, decoder_self_attention_k_cl = \
            self.bbox_head.transformer.decoder.layers[
                lid].forward_self_attention(query_te, query_cl,
                                            query_pos_te=query_pos_te,
                                            query_pos_cl=query_pos_cl)

            # 2. remaining layers in current decoder layer
            query_te = self.bbox_head.transformer.decoder.layers[
                lid].forward_remaining(query_te, key=None, value=memory_te,
                                       query_pos=query_pos_te,
                                       key_padding_mask=mask_flatten_te,
                                       reference_points=reference_points_input_te,
                                       spatial_shapes=spatial_shapes_te,
                                       level_start_index=level_start_index_te,
                                       valid_ratios=valid_ratios_te,
                                       reg_branches=reg_branches_te,
                                       **kwargs_te)
            query_cl = self.pts_bbox_head.transformer.decoder.layers[
                lid].forward_remaining(query_cl,
                                       key=key_cl,
                                       value=value_cl,
                                       query_pos=query_pos_cl,
                                       spatial_shapes=spatial_shapes_cl,
                                       level_start_index=level_start_index_cl,
                                       reference_points=reference_points_input_cl,
                                       **kwargs_cl)

            # 3. remaining operations (te)
            query_te = query_te.permute(1, 0, 2)
            if reg_branches_te is not None:
                tmp_te = reg_branches_te[lid](query_te)
                if reference_points_te.shape[-1] == 4:
                    new_reference_points_te = tmp_te + inverse_sigmoid(
                        reference_points_te)
                    new_reference_points_te = new_reference_points_te.sigmoid()
                else:
                    assert reference_points_te.shape[-1] == 2
                    new_reference_points_te = tmp_te
                    new_reference_points_te[..., :2] = tmp_te[..., :2] + inverse_sigmoid(reference_points_te)
                    new_reference_points_te = new_reference_points_te.sigmoid()
                reference_points_te = new_reference_points_te.detach()
            query_te = query_te.permute(1, 0, 2)
            if self.bbox_head.transformer.decoder.return_intermediate:
                intermediate_te.append(query_te)
                intermediate_reference_points_te.append(reference_points_te)

            # 3. remaining operations (cl)
            if self.pts_bbox_head.transformer.decoder.return_intermediate:
                intermediate_cl.append(query_cl)
                intermediate_reference_points_cl.append(reference_points_cl) # the each for each layer. check it. this is correct.

            # 4. TODO: (possibly) (intermediate) relation prediction using Q and K (check EGTR)

        # remaining operations (te)
        if self.bbox_head.transformer.decoder.return_intermediate:
            outputs_te_decoder = (torch.stack(intermediate_te), torch.stack(intermediate_reference_points_te))
        else:
            outputs_te_decoder = (query_te, reference_points_te)

        # remaining operations (cl)
        if self.pts_bbox_head.transformer.decoder.return_intermediate:
            outputs_cl_decoder = (torch.stack(intermediate_cl), torch.stack(intermediate_reference_points_cl))
        else:
            outputs_cl_decoder = (query_cl, reference_points_cl)

        # TODO: (possibly) final relation prediction using Q and K (check EGTR)

        # 4. go through the second half
        outputs_te_transformer_second_half = self.bbox_head.transformer.forward_second_half(*outputs_te_decoder, outputs_te_transformer_first_half['init_reference_out'])
        bbox_outs = self.bbox_head.forward_second_half(*outputs_te_transformer_second_half)
        outputs_cl_transformer_last_half = self.pts_bbox_head.transformer.forward_second_half(*outputs_cl_decoder, outputs_cl_transformer_first_half['init_reference_out'])
        outs = self.pts_bbox_head.forward_second_half(outputs_cl_transformer_last_half)

        # 5. TODO: loss computation
        te_losses = {}
        bbox_losses, te_assign_result = self.bbox_head.loss(bbox_outs, gt_bboxes, gt_labels, bbox_img_metas, gt_bboxes_ignore)
        for loss in bbox_losses:
            te_losses['bbox_head.' + loss] = bbox_losses[loss]
        num_gt_bboxes = sum([len(gt) for gt in gt_labels])
        if num_gt_bboxes == 0:
            for loss in te_losses:
                te_losses[loss] *= 0

        losses = dict()
        loss_inputs = [outs, gt_lanes_3d, gt_lane_labels_3d, te_assign_result]
        lane_losses = self.pts_bbox_head.my_loss(*loss_inputs, img_metas=img_metas)
        for loss in lane_losses:
            losses['lane_head.' + loss] = lane_losses[loss]

        losses.update(te_losses)


        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        new_prev_bev, results_list = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return results_list

    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None,
                        rescale=False):
        """Test function"""
        batchsize = len(img_metas)

        front_view_img_feats = [lvl[:, 0] for lvl in x]
        batch_input_shape = tuple(img[0, 0].size()[-2:])
        bbox_img_metas = []
        for img_meta in img_metas:
            bbox_img_metas.append(
                dict(
                    batch_input_shape=batch_input_shape,
                    img_shape=img_meta['img_shape'][0],
                    scale_factor=img_meta['scale_factor'][0],
                    crop_shape=img_meta['crop_shape'][0]))
            img_meta['batch_input_shape'] = batch_input_shape
        bbox_outs = self.bbox_head(front_view_img_feats, bbox_img_metas)
        bbox_results = self.bbox_head.get_bboxes(bbox_outs, bbox_img_metas,
                                                 rescale=rescale)
        te_feats = bbox_outs['history_states']
        te_cls_scores = bbox_outs['all_cls_scores']
        bev_feats = self.bev_constructor(x, img_metas, prev_bev)

        outs = self.pts_bbox_head(x, bev_feats, img_metas, te_feats,
                                  te_cls_scores)
        lane_results, lclc_results, lcte_results = self.pts_bbox_head.get_lanes(
            outs, img_metas, rescale=rescale)

        return bev_feats, bbox_results, lane_results, lclc_results, lcte_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        results_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_results, lane_results, lclc_results, lcte_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale)
        for result_dict, bbox, lane, lclc, lcte in zip(results_list,
                                                       bbox_results,
                                                       lane_results,
                                                       lclc_results,
                                                       lcte_results):
            result_dict['bbox_results'] = bbox
            result_dict['lane_results'] = lane
            result_dict['lclc_results'] = lclc
            result_dict['lcte_results'] = lcte
        return new_prev_bev, results_list
