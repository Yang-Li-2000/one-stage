#---------------------------------------------------------------------------------------#
# Graph-based Topology Reasoning for Driving Scenes (https://arxiv.org/abs/2304.05277)  #
# Source code: https://github.com/OpenDriveLab/TopoNet                                  #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss, build_head
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder

# for customized transformer (TopoNetTransformerDecoderOnly)
from mmdet.models.utils.builder import TRANSFORMER
from projects.toponet.models.modules.transformer_decoder_only import TopoNetTransformerDecoderOnly

# for customized decoder
from projects.toponet.models.modules.sgnn_decoder import TopoNetSGNNDecoder
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

# for customized decoder layer
from projects.toponet.models.modules.sgnn_decoder import SGNNDecoderLayer
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER

import warnings


@TRANSFORMER_LAYER.register_module()
class MySGNNDecoderLayer(SGNNDecoderLayer):

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super().__init__(attn_cfgs,
                         ffn_cfgs,
                         operation_order=operation_order,
                         norm_cfg=norm_cfg, **kwargs)

    def forward_self_attention(self,
                               query_te,
                               query_cl,
                               query_pos_te=None,
                               query_pos_cl=None,
                               attn_masks=None,
                               query_key_padding_mask=None):

        if attn_masks is not None or query_key_padding_mask is not None:
            raise NotImplementedError()

        if query_te is not None:
            raise NotImplementedError()
        else:
            query = query_cl
            query_pos = query_pos_cl

        identity = query

        attn_index = 0
        temp_key = temp_value = query

        query, decoder_self_attention_q, decoder_self_attention_k = \
            self.attentions[attn_index](
                query,
                temp_key,
                temp_value,
                identity if self.pre_norm else None,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=attn_masks,
                key_padding_mask=query_key_padding_mask,
            )

        # Split and return
        num_queries_te = 0

        query_te = query[:num_queries_te]
        query_cl = query[num_queries_te:]
        decoder_self_attention_q_te = decoder_self_attention_q[:num_queries_te]
        decoder_self_attention_q_cl = decoder_self_attention_q[num_queries_te:]
        decoder_self_attention_k_te = decoder_self_attention_k[:num_queries_te]
        decoder_self_attention_k_cl = decoder_self_attention_k[num_queries_te:]

        return query_te, query_cl, decoder_self_attention_q_te, decoder_self_attention_q_cl, decoder_self_attention_k_te, decoder_self_attention_k_cl

    def forward_remaining(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):

        norm_index = 0
        attn_index = 1
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'

        # check edge cases that are not implemented (when the nuber of self-attn is not 1)
        if self.operation_order.count('self_attn') < 1:
            raise NotImplementedError(
                'decoder layer not implemented when no self-attention exist')
        elif self.operation_order.count('self_attn') > 1:
            raise NotImplementedError(
                'decoder layer not implemented when multiple passes of self-attention exist')

        for layer in self.operation_order:
            if layer == 'self_attn':
               pass

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                # FFN_SGNN: linear layers + gnn layers
                query = self.ffns[ffn_index](
                    query, identity=identity if self.pre_norm else None)
                ffn_index += 1

        return query



    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                te_query=None,
                te_cls_scores=None,
                lclc_adj=None,
                lcte_adj=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        # check edge cases that are not implemented (when the nuber of self-attn is not 1)
        if self.operation_order.count('self_attn') < 1:
            raise NotImplementedError('decoder layer not implemented when no self-attention exist')
        elif self.operation_order.count('self_attn') > 1:
            raise NotImplementedError('decoder layer not implemented when multiple passes of self-attention exist')

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query, decoder_self_attention_q, decoder_self_attention_k = \
                self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
                # TODO

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                # FFN_SGNN: linear layers + gnn layers
                query = self.ffns[ffn_index](
                    query, te_query, lclc_adj, lcte_adj, te_cls_scores, identity=identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MyTopoNetSGNNDecoder(TopoNetSGNNDecoder):
    def __init__(self, *args, return_intermediate=False, **kwargs):

        super().__init__(*args, return_intermediate=return_intermediate, **kwargs)

    def forward(self,
                query,
                *args,
                reference_points=None,
                lclc_branches=None,
                lcte_branches=None,
                key_padding_mask=None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):

        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_lclc_rel = []
        intermediate_lcte_rel = []
        num_query = query.size(0)
        num_te_query = te_feats.size(2)

        prev_lclc_adj = torch.zeros((query.size(1), num_query, num_query),
                                  dtype=query.dtype, device=query.device)
        prev_lcte_adj = torch.zeros((query.size(1), num_query, num_te_query),
                                  dtype=query.dtype, device=query.device)
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2

            # TODO: use decoder_self_attention_q and decoder_self_attention_k
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                te_query=te_feats[lid],
                te_cls_scores=te_cls_scores[lid],
                lclc_adj=prev_lclc_adj,
                lcte_adj=prev_lcte_adj,
                **kwargs)
            output = output.permute(1, 0, 2)

            lclc_rel_out = lclc_branches[lid](output, output)
            lclc_rel_adj = lclc_rel_out.squeeze(-1).sigmoid()
            prev_lclc_adj = lclc_rel_adj.detach()

            lcte_rel_out = lcte_branches[lid](output, te_feats[lid])
            lcte_rel_adj = lcte_rel_out.squeeze(-1).sigmoid()
            prev_lcte_adj = lcte_rel_adj.detach()

            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_lclc_rel.append(lclc_rel_out)
                intermediate_lcte_rel.append(lcte_rel_out)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), torch.stack(
                intermediate_lclc_rel), torch.stack(
                intermediate_lcte_rel)

        return output, reference_points, lclc_rel_out, lcte_rel_out



@TRANSFORMER.register_module()
class MyTopoNetTransformerDecoderOnly(TopoNetTransformerDecoderOnly):

    def __init__(self,
                 decoder=None,
                 embed_dims=256,
                 pts_dim=3,
                 **kwargs):
        super(MyTopoNetTransformerDecoderOnly, self).__init__(
            decoder=decoder,
            embed_dims=embed_dims,
            pts_dim=pts_dim,
            **kwargs)

    @auto_fp16(apply_to=(
            'mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev',
            'bev_pos'))
    def forward_first_half(self,
                           mlvl_feats,
                           bev_embed,
                           object_query_embed,
                           bev_h,
                           bev_w,
                           lclc_branches=None,
                           lcte_branches=None,
                           te_feats=None,
                           te_cls_scores=None,
                           **kwargs):
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)


        # return variables
        query = query
        key = None
        value = bev_embed
        query_pos = query_pos
        reference_points = reference_points
        lclc_branches = lclc_branches
        lcte_branches = lcte_branches
        te_feats = te_feats
        te_cls_scores = te_cls_scores
        spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
        level_start_index = torch.tensor([0], device=query.device)

        output_dict = {"query": query,
                       "key": key,
                       "value": value,
                       "query_pos": query_pos,
                       "reference_points": reference_points,
                       "lclc_branches": lclc_branches,
                       "lcte_branches": lcte_branches,
                       "te_feats": te_feats,
                       "te_cls_scores": te_cls_scores,
                       "spatial_shapes": spatial_shapes,
                       "level_start_index": level_start_index,
                       "init_reference_out": init_reference_out,
                       "kwargs": kwargs}



        # TODO: note that kwargs is replaced with img_metas
        # return query, key, value, query_pos, reference_points, lclc_branches, lcte_branches, te_feats, te_cls_scores, spatial_shapes, level_start_index, img_metas, init_reference_out
        return output_dict



    @auto_fp16(apply_to=(
            'mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev',
            'bev_pos'))
    def forward_second_half(self, inter_states, inter_references, init_reference_out):

        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out


    @auto_fp16(apply_to=(
    'mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_embed,
                object_query_embed,
                bev_h,
                bev_w,
                lclc_branches=None,
                lcte_branches=None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        inter_states, inter_references, inter_lclc_rel, inter_lcte_rel = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            lclc_branches=lclc_branches,
            lcte_branches=lcte_branches,
            te_feats=te_feats,
            te_cls_scores=te_cls_scores,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out, inter_lclc_rel, inter_lcte_rel


@HEADS.register_module()
class TopoNetHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 transformer=None,
                 lclc_head=None,
                 lcte_head=None,
                 bbox_coder=None,
                 num_reg_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 pc_range=None,
                 pts_dim=3,
                 sync_cls_avg_factor=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.5),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.025),
                 train_cfg=dict(
                     assigner=dict(
                        type='LaneHungarianAssigner3D',
                        cls_cost=dict(type='FocalLossCost', weight=1.5),
                        reg_cost=dict(type='LaneL1Cost', weight=0.025))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'

            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        if lclc_head is not None:
            self.lclc_cfg = lclc_head

        if lcte_head is not None:
            self.lcte_cfg = lcte_head

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        assert pts_dim in (2, 3)
        self.pts_dim = pts_dim

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = pts_dim * 11
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, ] * self.code_size
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.gt_c_save = self.code_size

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_reg_fcs = num_reg_fcs
        self._init_layers()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        lclc_branch = build_head(self.lclc_cfg)
        lcte_branch = build_head(self.lcte_cfg)

        te_embed_branch = []
        in_channels = self.embed_dims
        for _ in range(self.num_reg_fcs - 1):
            te_embed_branch.append(nn.Sequential(
                    Linear(in_channels, 2 * self.embed_dims),
                    nn.ReLU(),
                    nn.Dropout(0.1)))
            in_channels = 2 * self.embed_dims
        te_embed_branch.append(Linear(2 * self.embed_dims, self.embed_dims))
        te_embed_branch = nn.Sequential(*te_embed_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.transformer.decoder.num_layers
        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.lclc_branches = _get_clones(lclc_branch, num_pred)
        self.lcte_branches = _get_clones(lcte_branch, num_pred)
        self.te_embed_branches = _get_clones(te_embed_branch, num_pred)

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward_first_half(self, mlvl_feats, bev_feats, img_metas, te_feats, te_cls_scores):
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)

        if te_feats is not None:
            te_feats = torch.stack([self.te_embed_branches[lid](te_feats[lid]) for lid in range(len(te_feats))])

        # return variables
        bev_h = self.bev_h
        bev_w = self.bev_w
        lclc_branches = self.lclc_branches
        lcte_branches = self.lcte_branches
        te_feats = te_feats
        te_cls_scores = te_cls_scores

        output_dict = {
            "mlvl_feats": mlvl_feats,
            "bev_feats": bev_feats,
            "object_query_embeds": object_query_embeds,
            "bev_h": bev_h,
            "bev_w": bev_w,
            "lclc_branches": lclc_branches,
            "lcte_branches": lcte_branches,
            "te_feats": te_feats,
            "te_cls_scores": te_cls_scores,
            "img_metas":img_metas
        }

        return output_dict

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward_second_half(self, outputs):

        hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == self.pts_dim

            bs, num_query, _ = tmp.shape
            tmp = tmp.view(bs, num_query, -1, self.pts_dim)
            tmp = tmp + reference.unsqueeze(2)
            tmp = tmp.sigmoid()

            coord = tmp.clone()
            coord[..., 0] = coord[..., 0] * (
                        self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1] = coord[..., 1] * (
                        self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if self.pts_dim == 3:
                coord[..., 2] = coord[..., 2] * (
                            self.pc_range[5] - self.pc_range[2]) + \
                                self.pc_range[2]
            outputs_coord = coord.view(bs, num_query, -1).contiguous()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'all_cls_scores': outputs_classes,
            'all_lanes_preds': outputs_coords,
            'history_states': hs
        }

        return outs


    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, bev_feats, img_metas, te_feats, te_cls_scores):

        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)

        te_feats = torch.stack([self.te_embed_branches[lid](te_feats[lid]) for lid in range(len(te_feats))])

        outputs = self.transformer(
            mlvl_feats,
            bev_feats,
            object_query_embeds,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            lclc_branches=self.lclc_branches,
            lcte_branches=self.lcte_branches,
            te_feats=te_feats,
            te_cls_scores=te_cls_scores,
            img_metas=img_metas,
        )

        hs, init_reference, inter_references, lclc_rel_out, lcte_rel_out = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == self.pts_dim

            bs, num_query, _ = tmp.shape
            tmp = tmp.view(bs, num_query, -1, self.pts_dim)
            tmp = tmp + reference.unsqueeze(2)
            tmp = tmp.sigmoid()

            coord = tmp.clone()
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if self.pts_dim == 3:
                coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outputs_coord = coord.view(bs, num_query, -1).contiguous()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'all_cls_scores': outputs_classes,
            'all_lanes_preds': outputs_coords,
            'all_lclc_preds': lclc_rel_out,
            'all_lcte_preds': lcte_rel_out,
            'history_states': hs
        }

        return outs

    def _my_get_target_single(self,
                           cls_score,
                           lanes_pred,
                           gt_labels,
                           gt_lanes,
                           gt_bboxes_ignore=None):

        num_bboxes = lanes_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(lanes_pred, cls_score, gt_lanes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, lanes_pred,
                                              gt_lanes)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lanes.new_full((num_bboxes,), self.num_classes,
                                   dtype=torch.long)
        labels[pos_inds] = gt_labels[
            sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_lanes.new_ones(num_bboxes)

        # bbox targets
        gt_c = gt_lanes.shape[-1]
        if gt_c == 0:
            gt_c = self.gt_c_save
            sampling_result.pos_gt_bboxes = torch.zeros((0, gt_c)).to(
                sampling_result.pos_gt_bboxes.device)
        else:
            self.gt_c_save = gt_c

        bbox_targets = torch.zeros_like(lanes_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(lanes_pred)
        bbox_weights[pos_inds] = 1.0
        # DETR

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes


        xs = pos_inds.unsqueeze(-1).repeat(1, pos_inds.size(0))
        ys = pos_inds.unsqueeze(0).repeat(pos_inds.size(0), 1)


        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, pos_assigned_gt_inds)

    def _get_target_single(self,
                           cls_score,
                           lanes_pred,
                           lclc_pred,
                           gt_labels,
                           gt_lanes,
                           gt_lane_adj,
                           gt_bboxes_ignore=None):

        num_bboxes = lanes_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(lanes_pred, cls_score, gt_lanes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, lanes_pred,
                                              gt_lanes)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lanes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_lanes.new_ones(num_bboxes)

        # bbox targets
        gt_c = gt_lanes.shape[-1]
        if gt_c == 0:
            gt_c = self.gt_c_save
            sampling_result.pos_gt_bboxes = torch.zeros((0, gt_c)).to(sampling_result.pos_gt_bboxes.device)
        else:
            self.gt_c_save = gt_c

        bbox_targets = torch.zeros_like(lanes_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(lanes_pred)
        bbox_weights[pos_inds] = 1.0
        # DETR

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        lclc_target = torch.zeros_like(lclc_pred.squeeze(-1), dtype=gt_lane_adj.dtype, device=lclc_pred.device)
        xs = pos_inds.unsqueeze(-1).repeat(1, pos_inds.size(0))
        ys = pos_inds.unsqueeze(0).repeat(pos_inds.size(0), 1)
        lclc_target[xs, ys] = gt_lane_adj[pos_assigned_gt_inds][:, pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights, lclc_target,
                pos_inds, neg_inds, pos_assigned_gt_inds)

    def my_get_targets(self,
                    cls_scores_list,
                    lanes_preds_list,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (
        labels_list, label_weights_list, lanes_targets_list, lanes_weights_list,
        pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._my_get_target_single, cls_scores_list, lanes_preds_list,
            gt_labels_list, gt_lanes_list,
            gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list,
            pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )
        return (
        labels_list, label_weights_list, lanes_targets_list, lanes_weights_list,
        num_total_pos, num_total_neg, assign_result)

    def get_targets(self,
                    cls_scores_list,
                    lanes_preds_list,
                    lclc_preds_list,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_lane_adj_list,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, lanes_targets_list, lanes_weights_list, lclc_targets_list,
            pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, lanes_preds_list, lclc_preds_list,
            gt_labels_list, gt_lanes_list, gt_lane_adj_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list, pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )
        return (labels_list, label_weights_list, lanes_targets_list, lanes_weights_list, lclc_targets_list,
                num_total_pos, num_total_neg, assign_result)


    def my_loss_single(self,
                    cls_scores,
                    lanes_preds,
                    te_assign_result,
                    gt_lanes_list,
                    gt_labels_list,
                    layer_index,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.my_get_targets(cls_scores_list, lanes_preds_list,
                                           gt_lanes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, assign_result) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))
        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan, :self.code_size],
            bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        # lclc loss
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox


    def my_loss_single_compatible_with_intermediate_relations(self,
                    cls_scores,
                    lanes_preds,
                    lclc_preds,
                    lcte_preds,
                    te_assign_result,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_lane_adj_list,
                    gt_lane_lcte_adj_list,
                    layer_index,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]
        lclc_preds_list = [lclc_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, lanes_preds_list, lclc_preds_list,
                                           gt_lanes_list, gt_labels_list, gt_lane_adj_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, lclc_targets_list,
         num_total_pos, num_total_neg, assign_result) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        lclc_targets = torch.cat(lclc_targets_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))
        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan, :self.code_size],
            bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        # lclc loss
        lclc_targets = 1 - lclc_targets.view(-1).long()
        lclc_preds = lclc_preds.view(-1, 1)
        loss_lclc = self.lclc_branches[-1].loss_rel(lclc_preds, lclc_targets)

        loss_lcte = self.lcte_branches[-1].loss(lcte_preds, gt_lane_lcte_adj_list, assign_result, te_assign_result)['loss_rel']

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, loss_lclc, loss_lcte



    def loss_single(self,
                    cls_scores,
                    lanes_preds,
                    lclc_preds,
                    lcte_preds,
                    te_assign_result,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_lane_adj_list,
                    gt_lane_lcte_adj_list,
                    layer_index,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]
        lclc_preds_list = [lclc_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, lanes_preds_list, lclc_preds_list, 
                                           gt_lanes_list, gt_labels_list, gt_lane_adj_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, lclc_targets_list,
         num_total_pos, num_total_neg, assign_result) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        lclc_targets = torch.cat(lclc_targets_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))
        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan, :self.code_size], 
            bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        # lclc loss
        lclc_targets = 1 - lclc_targets.view(-1).long()
        lclc_preds = lclc_preds.view(-1, 1)
        loss_lclc = self.lclc_branches[layer_index].loss_rel(lclc_preds, lclc_targets)

        loss_lcte = self.lcte_branches[layer_index].loss(lcte_preds, gt_lane_lcte_adj_list, assign_result, te_assign_result)['loss_rel']

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, loss_lclc, loss_lcte

    @force_fp32(apply_to=('preds_dicts'))
    def my_loss_with_intermediate_relations(self,
                preds_dicts,
                gt_lanes_3d,
                gt_labels_list,
                gt_lane_adj,
                gt_lane_lcte_adj,
                te_assign_results,
                gt_bboxes_ignore=None,
                img_metas=None,
                pred_connectivity_tecl=None,
                pred_connectivity_clcl=None):

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        all_cls_scores = preds_dicts['all_cls_scores']
        all_lanes_preds = preds_dicts['all_lanes_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_lanes_list = [lane for lane in gt_lanes_3d]

        all_gt_lanes_list = [gt_lanes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        layer_index = [i for i in range(num_dec_layers)]

        loss_dict = dict()

        # TODO: check the relationship between te_assign_results and model.pts_bbox_head.te_embed_branches
        if True:
            # losses_cls_final, losses_bbox_final, losses_lclc_final, losses_lcte_final = self.loss_single(
            #     all_cls_scores[-1],
            #     all_lanes_preds[-1],
            #     pred_connectivity_clcl,
            #     pred_connectivity_tecl,
            #     te_assign_results[-1],
            #     all_gt_lanes_list[-1],
            #     all_gt_labels_list[-1],
            #     gt_lane_adj,
            #     gt_lane_lcte_adj,
            #     layer_index[-1],
            # )
            all_gt_lane_adj_list = [gt_lane_adj for _ in range(num_dec_layers)]
            all_gt_lane_lcte_adj_list = [gt_lane_lcte_adj for _ in range(num_dec_layers)]

            losses_cls, losses_bbox, losses_lclc, losses_lcte = multi_apply(
                self.my_loss_single_compatible_with_intermediate_relations, all_cls_scores, all_lanes_preds,
                pred_connectivity_clcl, pred_connectivity_tecl, te_assign_results,
                all_gt_lanes_list, all_gt_labels_list, all_gt_lane_adj_list,
                all_gt_lane_lcte_adj_list, layer_index)

        # loss from the last decoder layer
        loss_dict['loss_lane_cls'] = losses_cls[-1]
        loss_dict['loss_lane_reg'] = losses_bbox[-1]
        loss_dict['loss_lclc_rel'] = losses_lclc[-1]
        loss_dict['loss_lcte_rel'] = losses_lcte[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_lclc_i, loss_lcte_i in zip(
            losses_cls[:-1], losses_bbox[:-1], losses_lclc[:-1], losses_lcte[:-1]):

            loss_dict[f'd{num_dec_layer}.loss_lane_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_lane_reg'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_lclc_rel'] = loss_lclc_i
            loss_dict[f'd{num_dec_layer}.loss_lcte_rel'] = loss_lcte_i
            num_dec_layer += 1
        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def my_loss(self,
                preds_dicts,
                gt_lanes_3d,
                gt_labels_list,
                gt_lane_adj,
                gt_lane_lcte_adj,
                te_assign_results,
                gt_bboxes_ignore=None,
                img_metas=None,
                pred_connectivity_tecl=None,
                pred_connectivity_clcl=None):

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        all_cls_scores = preds_dicts['all_cls_scores']
        all_lanes_preds = preds_dicts['all_lanes_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_lanes_list = [lane for lane in gt_lanes_3d]

        all_gt_lanes_list = [gt_lanes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        layer_index = [i for i in range(num_dec_layers)]

        SEPARATE_LAST_LAYER_LOSS = pred_connectivity_tecl is not None
        if SEPARATE_LAST_LAYER_LOSS:
            num_dec_layers -= 1

        losses_cls, losses_bbox = multi_apply(
            self.my_loss_single,
            all_cls_scores[:num_dec_layers],
            all_lanes_preds[:num_dec_layers],
            te_assign_results[:num_dec_layers],
            all_gt_lanes_list[:num_dec_layers],
            all_gt_labels_list[:num_dec_layers],
            layer_index[:num_dec_layers])

        loss_dict = dict()

        # TODO: check the relationship between te_assign_results and model.pts_bbox_head.te_embed_branches
        if SEPARATE_LAST_LAYER_LOSS:
            losses_cls_final, losses_bbox_final, losses_lclc_final, losses_lcte_final = self.loss_single(
                all_cls_scores[-1],
                all_lanes_preds[-1],
                pred_connectivity_clcl,
                pred_connectivity_tecl,
                te_assign_results[-1],
                all_gt_lanes_list[-1],
                all_gt_labels_list[-1],
                gt_lane_adj,
                gt_lane_lcte_adj,
                layer_index[-1],
            )
            # add losses from the last layer
            losses_cls.append(losses_cls_final)
            losses_bbox.append(losses_bbox_final)
            loss_dict['loss_lclc_rel'] = losses_lclc_final
            loss_dict['loss_lcte_rel'] = losses_lcte_final

        # loss from the last decoder layer
        loss_dict['loss_lane_cls'] = losses_cls[-1]
        loss_dict['loss_lane_reg'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(
                losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_lane_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_lane_reg'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             gt_lanes_3d,
             gt_labels_list,
             gt_lane_adj,
             gt_lane_lcte_adj,
             te_assign_results,
             gt_bboxes_ignore=None,
             img_metas=None):

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        all_cls_scores = preds_dicts['all_cls_scores']
        all_lanes_preds = preds_dicts['all_lanes_preds']
        all_lclc_preds = preds_dicts['all_lclc_preds']
        all_lcte_preds = preds_dicts['all_lcte_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_lanes_list = [lane for lane in gt_lanes_3d]

        all_gt_lanes_list = [gt_lanes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_lane_adj_list = [gt_lane_adj for _ in range(num_dec_layers)]
        all_gt_lane_lcte_adj_list = [gt_lane_lcte_adj for _ in range(num_dec_layers)]
        layer_index = [i for i in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_lclc, losses_lcte = multi_apply(
            self.loss_single, all_cls_scores, all_lanes_preds, all_lclc_preds, all_lcte_preds, te_assign_results,
            all_gt_lanes_list, all_gt_labels_list, all_gt_lane_adj_list, all_gt_lane_lcte_adj_list, layer_index)

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_lane_cls'] = losses_cls[-1]
        loss_dict['loss_lane_reg'] = losses_bbox[-1]
        loss_dict['loss_lclc_rel'] = losses_lclc[-1]
        loss_dict['loss_lcte_rel'] = losses_lcte[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_lclc_i, loss_lcte_i in zip(
            losses_cls[:-1], losses_bbox[:-1], losses_lclc[:-1], losses_lcte[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_lane_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_lane_reg'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_lclc_rel'] = loss_lclc_i
            loss_dict[f'd{num_dec_layer}.loss_lcte_rel'] = loss_lcte_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, rescale=False):

        all_lclc_preds = preds_dicts['all_lclc_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lclc_preds = [_ for _ in all_lclc_preds]

        all_lcte_preds = preds_dicts['all_lcte_preds'][-1].squeeze(-1).sigmoid().detach().cpu().numpy()
        all_lcte_preds = [_ for _ in all_lcte_preds]

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            lanes = preds['lane3d']
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([lanes, scores, labels])
        return ret_list, all_lclc_preds, all_lcte_preds
