# ---------------------------------------------------------------------------------------#
# Graph-based Topology Reasoning for Driving Scenes (https://arxiv.org/abs/2304.05277)  #
# Source code: https://github.com/OpenDriveLab/TopoNet                                  #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
# ---------------------------------------------------------------------------------------#

import time
import copy
import numpy as np
import torch
from torch import nn

from . import counts

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet3d.models.builder import build_neck
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from ...utils.builder import build_bev_constructor
from mmdet.models.utils.transformer import inverse_sigmoid

from .model.deformable_detr import (
    DeformableDetrHungarianMatcher,
    DeformableDetrMLPPredictionHead,
    DeformableDetrModel,
    DeformableDetrPreTrainedModel,
    inverse_sigmoid,
)

from . import MergedTopoNetMapGraph
from mmdet.models import DETECTORS, build_loss
from mmdet.models.builder import build_head, build_backbone, build_neck

@DETECTORS.register_module()
class MergedDistillTopoNetMapGraph(MergedTopoNetMapGraph):

    def __init__(self,
                 use_distill=None,
                 teacher_checkpoint=None,
                 loss_simple_bev=None,
                 student_img_backbone=None,
                 student_img_neck=None,
                 student_bev_constructor=None,
                 student_bbox_head=None,
                 student_lane_head=None,
                 **kwargs):

        super(MergedDistillTopoNetMapGraph, self).__init__(**kwargs)

        assert use_distill == True
        assert teacher_checkpoint is not None
        assert loss_simple_bev is not None

        self.use_distill = use_distill

        if student_img_backbone is not None:
            self.student_img_backbone = build_backbone(student_img_backbone)

        if student_img_neck is not None:
            self.student_img_neck = build_neck(student_img_neck)

        if student_bev_constructor is not None:
            self.student_bev_constructor = build_bev_constructor(student_bev_constructor)

        if student_bbox_head is not None:
            student_bbox_head.update(train_cfg=self.train_cfg.bbox)
            self.student_bbox_head = build_head(student_bbox_head)
        else:
            self.student_bbox_head = None

        if student_lane_head is not None:
            student_lane_head.update(train_cfg=self.train_cfg.lane)
            self.student_pts_bbox_head = build_head(student_lane_head)
        else:
            self.student_pts_bbox_head = None

        self.simple_bev_loss = build_loss(loss_simple_bev)

        teacher_state_dict = torch.load(teacher_checkpoint)
        self.load_state_dict(teacher_state_dict['state_dict'], strict=False)
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        for param in self.img_neck.parameters():
            param.requires_grad = False
        for param in self.bev_constructor.parameters():
            param.requires_grad = False
        for param in self.bbox_head.parameters():
            param.requires_grad = False
        for param in self.pts_bbox_head.parameters():
            param.requires_grad = False

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        assert self.use_distill == True
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            img_feats = self.img_backbone(img)
            student_img_feats = self.student_img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            if isinstance(student_img_feats, dict):
                student_img_feats = list(student_img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            student_img_feats = self.student_img_neck(student_img_feats)

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

        student_img_feats_reshaped = []
        for student_img_feat in student_img_feats:
            BN, C, H, W = student_img_feat.size()
            if len_queue is not None:
                student_img_feats_reshaped.append(
                    student_img_feat.view(int(B / len_queue), len_queue, int(BN / B), C,
                                          H, W))
            else:
                student_img_feats_reshaped.append(
                    student_img_feat.view(B, int(BN / B), C, H, W))

        return student_img_feats_reshaped, img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        student_img_feats, img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return student_img_feats, img_feats

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


    #Copied from merged_toponet.py
    def student_predict(self, img_feats, img_metas, prev_bev, front_view_img_feats, bbox_img_metas):

        # 1. prepare inputs
        te_feats = None
        te_cls_scores = None

        # TODO: separate bev feats for teacher and student
        bev_feats = self.student_bev_constructor(img_feats, img_metas, prev_bev)

        # 2. go through the first half
        outputs_te_head_first_half = self.student_bbox_head.forward_first_half(front_view_img_feats, bbox_img_metas)
        outputs_te_transformer_first_half = self.student_bbox_head.transformer.forward_first_half(*outputs_te_head_first_half)
        outputs_cl_head_first_half = self.student_pts_bbox_head.forward_first_half(img_feats, bev_feats, img_metas, te_feats, te_cls_scores)
        outputs_cl_transformer_first_half = self.student_pts_bbox_head.transformer.forward_first_half(
            outputs_cl_head_first_half['mlvl_feats'],
            outputs_cl_head_first_half['bev_feats'],
            outputs_cl_head_first_half['object_query_embeds'],
            outputs_cl_head_first_half['bev_h'],
            outputs_cl_head_first_half['bev_w'],
            img_metas=outputs_cl_head_first_half['img_metas']
        )

        # 3. manually go through decoders
        num_decoder_layers = self.num_decoder_layers

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

        # for relation prediction
        decoder_attention_queries_te = []
        decoder_attention_keys_te = []
        decoder_attention_queries_cl = []
        decoder_attention_keys_cl = []

        # looping over each decoder layer
        for lid in range(num_decoder_layers):

            # 0. input preparation (te)
            if reference_points_te.shape[-1] == 4:
                reference_points_input_te = reference_points_te[:, :, None] * torch.cat(
                    [valid_ratios_te, valid_ratios_te], -1)[:, None]
            else:
                assert reference_points_te.shape[-1] == 2
                reference_points_input_te = reference_points_te[:, :, None] * valid_ratios_te[:, None]

            # 0. input preparation (cl)
            reference_points_input_cl = reference_points_cl[..., :2].unsqueeze(2)

            # 1. self-attention after concatenating queries
            # te
            query_te, _, decoder_self_attention_q_te, _, decoder_self_attention_k_te, _ = \
                self.student_bbox_head.transformer.decoder.layers[
                    lid].forward_self_attention(query_te, None,
                                                query_pos_te=query_pos_te,
                                                query_pos_cl=None)
            # cl
            _, query_cl, _, decoder_self_attention_q_cl, _, decoder_self_attention_k_cl = \
                self.student_pts_bbox_head.transformer.decoder.layers[
                    lid].forward_self_attention(None, query_cl,
                                                query_pos_te=None,
                                                query_pos_cl=query_pos_cl)

            # Store q and k of te and cl in different lists
            decoder_attention_queries_te.append(decoder_self_attention_q_te)
            decoder_attention_keys_te.append(decoder_self_attention_k_te)
            decoder_attention_queries_cl.append(decoder_self_attention_q_cl)
            decoder_attention_keys_cl.append(decoder_self_attention_k_cl)

            # 2. remaining layers in current decoder layer
            query_te = self.student_bbox_head.transformer.decoder.layers[
                lid].forward_remaining(query_te, key=None, value=memory_te,
                                       query_pos=query_pos_te,
                                       key_padding_mask=mask_flatten_te,
                                       reference_points=reference_points_input_te,
                                       spatial_shapes=spatial_shapes_te,
                                       level_start_index=level_start_index_te,
                                       valid_ratios=valid_ratios_te,
                                       reg_branches=reg_branches_te,
                                       **kwargs_te)
            query_cl = self.student_pts_bbox_head.transformer.decoder.layers[
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
            if self.student_bbox_head.transformer.decoder.return_intermediate:
                intermediate_te.append(query_te)
                intermediate_reference_points_te.append(reference_points_te)

            # 3. remaining operations (cl)
            if self.student_pts_bbox_head.transformer.decoder.return_intermediate:
                intermediate_cl.append(query_cl)
                intermediate_reference_points_cl.append(
                    reference_points_cl)  # the each for each layer. check it. this is correct.

        # remaining operations (te)
        if self.student_bbox_head.transformer.decoder.return_intermediate:
            outputs_te_decoder = (torch.stack(intermediate_te), torch.stack(intermediate_reference_points_te))
        else:
            outputs_te_decoder = (query_te, reference_points_te)

        # remaining operations (cl)
        if self.student_pts_bbox_head.transformer.decoder.return_intermediate:
            outputs_cl_decoder = (torch.stack(intermediate_cl), torch.stack(intermediate_reference_points_cl))
        else:
            outputs_cl_decoder = (query_cl, reference_points_cl)

        # Relation prediction using Q and K
        bsz = reference_points_te.shape[0]
        num_object_queries = self.num_object_queries
        unscaling = self.head_dim ** 0.5

        # Unscaling & stacking attention queries
        device = decoder_attention_queries_te[0].device
        # For TE
        QK_te_shape = [6, 100, 1, 128]
        projected_q_te = torch.empty(QK_te_shape).to(device)  # Allocate once
        for i, (q, proj_q) in enumerate(zip(decoder_attention_queries_te, self.proj_q_te)):
            projected_q_te[i] = proj_q(q * unscaling)
        decoder_attention_queries_te = projected_q_te.permute(2, 1, 0, 3)
        del projected_q_te

        projected_k_te = torch.empty(QK_te_shape).to(device)
        for i, (k, proj_k) in enumerate(zip(decoder_attention_keys_te, self.proj_k_te)):
            projected_k_te[i] = proj_k(k)
        decoder_attention_keys_te = projected_k_te.permute(2, 1, 0, 3)
        del projected_k_te

        # For CL
        QK_cl_shape = [6, 200, 1, 128]
        projected_q_cl = torch.empty(QK_cl_shape).to(device)
        for i, (q, proj_q) in enumerate(zip(decoder_attention_queries_cl, self.proj_q_cl)):
            projected_q_cl[i] = proj_q(q * unscaling)
        decoder_attention_queries_cl = projected_q_cl.permute(2, 1, 0, 3)
        del projected_q_cl

        projected_k_cl = torch.empty(QK_cl_shape).to(device)
        for i, (k, proj_k) in enumerate(zip(decoder_attention_keys_cl, self.proj_k_cl)):
            projected_k_cl[i] = proj_k(k)
        decoder_attention_keys_cl = projected_k_cl.permute(2, 1, 0, 3)
        del projected_k_cl

        # concat before pairwise concat
        decoder_attention_queries = torch.cat([decoder_attention_queries_te, decoder_attention_queries_cl], dim=1)
        decoder_attention_keys = torch.cat([decoder_attention_keys_te, decoder_attention_keys_cl], dim=1)

        # Pairwise concatenation
        decoder_attention_queries = decoder_attention_queries.unsqueeze(2).expand(
            -1, -1, num_object_queries, -1, -1
        )
        decoder_attention_keys = decoder_attention_keys.unsqueeze(1).expand(
            -1, num_object_queries, -1, -1, -1
        )
        relation_source = torch.cat(
            [decoder_attention_queries, decoder_attention_keys],
            dim=-1
        )  # [bsz, num_object_queries, num_object_queries, num_layers, 2*d_model]
        del decoder_attention_queries, decoder_attention_keys

        # add final hidden represetations separatly for tecl and clcl.
        #  Specifically, clcl should only receive query_cl while tecl receives both
        # clcl
        sequence_output_clcl = query_cl.permute(1, 0, 2)
        subject_output_clcl = self.final_sub_proj_clcl(sequence_output_clcl).unsqueeze(2).expand(-1, -1, self.nq_cl, -1)
        object_output_clcl = self.final_obj_proj_clcl(sequence_output_clcl).unsqueeze(1).expand(-1, self.nq_cl, -1, -1)
        del sequence_output_clcl
        relation_source_clcl = torch.cat(
            [
                relation_source[:, -self.nq_cl:, -self.nq_cl:],
                torch.cat([subject_output_clcl, object_output_clcl], dim=-1).unsqueeze(
                    -2),
            ],
            dim=-2,
        )
        del subject_output_clcl, object_output_clcl

        # tecl
        sequence_output_tecl = torch.cat([query_te, query_cl], dim=0).permute(1, 0, 2)
        subject_output_tecl = self.final_sub_proj_tecl(sequence_output_tecl).unsqueeze(2).expand(-1, -1,
                                                                                                 num_object_queries, -1)
        object_output_tecl = self.final_obj_proj_tecl(sequence_output_tecl).unsqueeze(1).expand(-1, num_object_queries,
                                                                                                -1, -1)
        del sequence_output_tecl
        relation_source_tecl = torch.cat(
            [
                relation_source[:, :, :],
                torch.cat([subject_output_tecl, object_output_tecl], dim=-1).unsqueeze(
                    -2),
            ],
            dim=-2,
        )
        del subject_output_tecl, object_output_tecl

        # Gated sum
        relation_source_tecl = relation_source_tecl[:, self.nq_te:, :self.nq_te]
        rel_gate_tecl = self.rel_predictor_gate_tecl(relation_source_tecl).sigmoid_()
        gated_relation_source_tecl = (rel_gate_tecl * relation_source_tecl).sum(dim=-2)

        rel_gate_clcl = self.rel_predictor_gate_clcl(relation_source_clcl).sigmoid_()
        gated_relation_source_clcl = (rel_gate_clcl * relation_source_clcl).sum(dim=-2)

        # Connectivity
        pred_connectivity_tecl = self.connectivity_layer_tecl(gated_relation_source_tecl)
        pred_connectivity_clcl = self.connectivity_layer_clcl(gated_relation_source_clcl)
        del relation_source_tecl, relation_source_clcl, gated_relation_source_tecl, gated_relation_source_clcl
        del rel_gate_tecl, rel_gate_clcl
        del relation_source

        # 4. go through the second half
        outputs_te_transformer_second_half = self.student_bbox_head.transformer.forward_second_half(*outputs_te_decoder,
                                                                                            outputs_te_transformer_first_half[
                                                                                                'init_reference_out'])
        bbox_outs = self.student_bbox_head.forward_second_half(*outputs_te_transformer_second_half)
        outputs_cl_transformer_last_half = self.student_pts_bbox_head.transformer.forward_second_half(*outputs_cl_decoder,
                                                                                              outputs_cl_transformer_first_half[
                                                                                                  'init_reference_out'])
        outs = self.student_pts_bbox_head.forward_second_half(outputs_cl_transformer_last_half)

        outs['all_lclc_preds'] = [pred_connectivity_clcl]
        outs['all_lcte_preds'] = [pred_connectivity_tecl]

        return bbox_outs, outs, bev_feats

    # TDDO
    @auto_fp16(apply_to=('img'))
    def forward_train(self,
                      img=None,
                      map_graph=None,
                      img_metas=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_lanes_3d=None,
                      gt_lane_labels_3d=None,
                      gt_lane_adj=None,
                      gt_lane_lcte_adj=None,
                      gt_bboxes_ignore=None,
                      **kwargs
                      ):

        self.img_backbone.eval()
        self.img_neck.eval()
        self.bev_constructor.eval()
        self.bbox_head.eval()
        self.pts_bbox_head.eval()

        # 1. Generate inputs
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        else:
            prev_bev = None

        img_metas = [each[len_queue - 1] for each in img_metas]
        student_img_feats, img_feats = self.extract_feat(img=img, img_metas=img_metas)

        front_view_img_feats = [lvl[:, 0] for lvl in img_feats]
        student_front_view_img_feats = [lvl[:, 0] for lvl in student_img_feats]

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

        # 2. Generate predictions
        bbox_outs, outs, bev_feats = self.predict(img_feats, img_metas, prev_bev, front_view_img_feats, bbox_img_metas, map_graph=map_graph, **kwargs)
        student_bbox_outs, student_outs, student_bev_feats = self.student_predict(student_img_feats, img_metas, prev_bev, student_front_view_img_feats, bbox_img_metas)

        # 3. Compute Losses
        distill_losses = dict()
        simple_bev_feats_losses = self.distill_loss(bev_feats, student_bev_feats)
        for loss in simple_bev_feats_losses:
            distill_losses['distill.' + loss] = simple_bev_feats_losses[loss]

        te_losses = {}
        bbox_losses, te_assign_result = self.student_bbox_head.loss(student_bbox_outs, gt_bboxes, gt_labels, bbox_img_metas, gt_bboxes_ignore)
        for loss in bbox_losses:
            te_losses['bbox_head.' + loss] = bbox_losses[loss]
        num_gt_bboxes = sum([len(gt) for gt in gt_labels])
        if num_gt_bboxes == 0:
            for loss in te_losses:
                te_losses[loss] *= 0

        losses = dict()
        loss_inputs = [student_outs, gt_lanes_3d, gt_lane_labels_3d, gt_lane_adj, gt_lane_lcte_adj, te_assign_result]

        lane_losses = self.student_pts_bbox_head.my_loss(*loss_inputs, img_metas=img_metas, pred_connectivity_tecl=student_outs['all_lcte_preds'][-1], pred_connectivity_clcl=student_outs['all_lclc_preds'][-1])

        for loss in lane_losses:
            losses['lane_head.' + loss] = lane_losses[loss]

        losses.update(te_losses)
        losses.update(distill_losses)

        return losses

    # def simple_test_pts(self, x, map_x, img_metas, img=None, prev_bev=None,
    #                     rescale=False):
    #
    #     # 1. Generate inputs
    #     batchsize = len(img_metas)
    #     front_view_img_feats = [lvl[:, 0] for lvl in x]
    #     batch_input_shape = tuple(img[0, 0].size()[-2:])
    #     bbox_img_metas = []
    #     for img_meta in img_metas:
    #         bbox_img_metas.append(
    #             dict(
    #                 batch_input_shape=batch_input_shape,
    #                 img_shape=img_meta['img_shape'][0],
    #                 scale_factor=img_meta['scale_factor'][0],
    #                 crop_shape=img_meta['crop_shape'][0]))
    #         img_meta['batch_input_shape'] = batch_input_shape
    #
    #     # 2. Generate predictions
    #     bbox_outs, outs, bev_feats = self.student_predict(x, img_metas, prev_bev, front_view_img_feats, bbox_img_metas)
    #
    #     # 3. Get boxes, lanes, and relations
    #     bbox_results = self.student_bbox_head.get_bboxes(bbox_outs, bbox_img_metas, rescale=rescale)
    #     lane_results, lclc_results, lcte_results = self.student_pts_bbox_head.get_lanes(outs, img_metas, rescale=rescale)
    #
    #     return bev_feats, bbox_results, lane_results, lclc_results, lcte_results


    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None,
                        rescale=False):
        # 1. Generate inputs
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

        # 2. Generate predictions
        bbox_outs, outs, bev_feats = self.student_predict(x, img_metas, prev_bev, front_view_img_feats, bbox_img_metas)

        # 3. Get boxes, lanes, and relations
        bbox_results = self.student_bbox_head.get_bboxes(bbox_outs, bbox_img_metas, rescale=rescale)
        lane_results, lclc_results, lcte_results = self.student_pts_bbox_head.get_lanes(outs, img_metas, rescale=rescale)

        return bev_feats, bbox_results, lane_results, lclc_results, lcte_results

    def simple_test(self, img_metas, img=None, map_graph=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_feats, _ = self.extract_feat(img=img, img_metas=img_metas)

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

    def distill_loss(self,
                    bev_feats,
                    student_bev_feats):
        bevfeats_losses = dict()
        bevfeats_losses['origin_bev'] = self.simple_bev_loss(student_bev_feats, bev_feats)
        return bevfeats_losses