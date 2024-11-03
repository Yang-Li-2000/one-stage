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

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
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

from mmcv.cnn.bricks.transformer import MultiheadAttention
import torch.nn.functional as F

# Copied from HRMapNet/projects/mmdet3d_plugin/hrmap/global_map.py
from scipy.spatial.transform import Rotation as R

# Copied from HRMapNet/projects/mmdet3d_plugin/hrmap/global_map.py
def gen_matrix(ego2global_rotation, ego2global_translation):
    rotation_xyz = np.roll(ego2global_rotation, shift=-1)
    trans = np.eye(4)
    trans[:3, 3] = ego2global_translation
    trans[:3, :3] = R.from_quat(rotation_xyz).as_matrix()
    return trans

# Copied from HRMapNet/projects/mmdet3d_plugin/hrmap/global_map.py
def get_bev_coords(bev_bound_w, bev_bound_h, bev_w, bev_h):
    """
    Args:
        bev_bound_w (tuple:2):
        bev_bound_h (tuple:2):
        bev_w (int):
        bev_h (int):
    Returns: (bev_h, bev_w, 4)

    """
    sample_coords = torch.stack(torch.meshgrid(
        torch.linspace(bev_bound_w[0], bev_bound_w[1], int(bev_w), dtype=torch.float32),
        torch.linspace(bev_bound_h[0], bev_bound_h[1], int(bev_h), dtype=torch.float32)
    ), axis=2).transpose(1, 0)
    assert sample_coords.shape[0] == bev_h, sample_coords.shape[1] == bev_w
    zeros = torch.zeros((bev_h, bev_w, 1), dtype=sample_coords.dtype)
    ones = torch.ones((bev_h, bev_w, 1), dtype=sample_coords.dtype)
    sample_coords = torch.cat([sample_coords, zeros, ones], dim=-1)
    return sample_coords

# Copied from HRMapNet/projects/mmdet3d_plugin/hrmap/global_map.py
class GlobalMap:
    def __init__(self, map_cfg):
        self.map_type = torch.uint8
        self.fuse_method_val = map_cfg['fuse_method']
        self.fuse_method = map_cfg['fuse_method']
        self.bev_h = map_cfg['bev_h']
        self.bev_w = map_cfg['bev_w']
        self.pc_range = map_cfg['pc_range']
        self.load_map_path = map_cfg['load_map_path']
        self.save_map_path = map_cfg['save_map_path']
        self.bev_patch_h = self.pc_range[4] - self.pc_range[1]
        self.bev_patch_w = self.pc_range[3] - self.pc_range[0]
        bev_radius = np.sqrt(self.bev_patch_h ** 2 + self.bev_patch_w ** 2) / 2
        dataset = map_cfg['dataset']
        if dataset == 'av2':
            self.city_list = ['WDC', 'MIA', 'PAO', 'PIT', 'ATX', 'DTW']
            bev_radius = bev_radius * 5
            self.train_min_lidar_loc = {'WDC': np.array([2327.78751629, 25.76974403]) - bev_radius,
                                        'MIA': np.array([-1086.92985063, -464.15366362]) - bev_radius,
                                        'PAO': np.array([-2225.36229607, -309.44287914]) - bev_radius,
                                        'PIT': np.array([695.66044791, -443.89844576]) - bev_radius,
                                        'ATX': np.array([589.98724063, -2444.36667873]) - bev_radius,
                                        'DTW': np.array([-6111.0784155, 628.12019426]) - bev_radius}
            self.train_max_lidar_loc = {'WDC': np.array([6951.32050819, 4510.96637507]) + bev_radius,
                                        'MIA': np.array([6817.02338386, 4301.35442342]) + bev_radius,
                                        'PAO': np.array([1646.099298, 2371.23617712]) + bev_radius,
                                        'PIT': np.array([7371.45409948, 3314.83461676]) + bev_radius,
                                        'ATX': np.array([3923.01840213, -1161.67712224]) + bev_radius,
                                        'DTW': np.array([11126.80825267, 6045.01530619]) + bev_radius}
            self.val_min_lidar_loc = {'WDC': np.array([1664.20793519, 344.29333819]) - bev_radius,
                                      'MIA': np.array([-885.96340492, 257.79835061]) - bev_radius,
                                      'PAO': np.array([-3050.01628955, -18.25448306]) - bev_radius,
                                      'PIT': np.array([715.98981458, -136.13570664]) - bev_radius,
                                      'ATX': np.array([840.66655697, -2581.61138577]) - bev_radius,
                                      'DTW': np.array([36.60503836, 2432.04117045]) - bev_radius}
            self.val_max_lidar_loc = {'WDC': np.array([6383.48765357, 4320.74293797]) + bev_radius,
                                      'MIA': np.array([6708.79270643, 4295.23306249]) + bev_radius,
                                      'PAO': np.array([654.02351246, 2988.66862304]) + bev_radius,
                                      'PIT': np.array([7445.46486881, 3160.2406237]) + bev_radius,
                                      'ATX': np.array([3726.62166299, -1296.12914951]) + bev_radius,
                                      'DTW': np.array([10896.30840694, 6215.31771939]) + bev_radius}
        else:
            self.city_list = ['singapore-onenorth', 'boston-seaport',
                              'singapore-queenstown', 'singapore-hollandvillage']
            self.train_min_lidar_loc = {'singapore-onenorth': np.array([118., 419.]) - bev_radius,
                                      'boston-seaport': np.array([298., 328.]) - bev_radius,
                                      'singapore-queenstown': np.array([347., 862.]) - bev_radius,
                                      'singapore-hollandvillage': np.array([442., 902.]) - bev_radius}
            self.train_max_lidar_loc = {'singapore-onenorth': np.array([1232., 1777.]) + bev_radius,
                                      'boston-seaport': np.array([2527., 1896.]) + bev_radius,
                                      'singapore-queenstown': np.array([2685., 3298.]) + bev_radius,
                                      'singapore-hollandvillage': np.array([2490., 2839.]) + bev_radius}
            self.val_min_lidar_loc = {'singapore-onenorth': np.array([118., 409.]) - bev_radius,
                                    'boston-seaport': np.array([411., 554.]) - bev_radius,
                                    'singapore-queenstown': np.array([524., 870.]) - bev_radius,
                                    'singapore-hollandvillage': np.array([608., 2006.]) - bev_radius}
            self.val_max_lidar_loc = {'singapore-onenorth': np.array([1232., 1732.]) + bev_radius,
                                    'boston-seaport': np.array([2368., 1720.]) + bev_radius,
                                    'singapore-queenstown': np.array([2043., 3334.]) + bev_radius,
                                    'singapore-hollandvillage': np.array([2460., 2836.]) + bev_radius}
            self.mix_min_lidar_loc = {'singapore-onenorth': np.array([118., 409.]) - bev_radius,
                                      'boston-seaport': np.array([298., 328.]) - bev_radius,
                                      'singapore-queenstown': np.array([347., 862.]) - bev_radius,
                                      'singapore-hollandvillage': np.array([442., 902.]) - bev_radius}
            self.mix_max_lidar_loc = {'singapore-onenorth': np.array([1232., 1777.]) + bev_radius,
                                      'boston-seaport': np.array([2527., 1896.]) + bev_radius,
                                      'singapore-queenstown': np.array([2685., 3334.]) + bev_radius,
                                      'singapore-hollandvillage': np.array([2490., 2839.]) + bev_radius}

        bev_bound_h, bev_bound_w = \
            [(-row[0] / 2 + row[0] / row[1] / 2, row[0] / 2 - row[0] / row[1] / 2)
             for row in ((self.bev_patch_h, self.bev_h), (self.bev_patch_w, self.bev_w))]
        self.bev_grid_len_h = self.bev_patch_h / self.bev_h
        self.bev_grid_len_w = self.bev_patch_w / self.bev_w
        self.bev_coords = get_bev_coords(bev_bound_w, bev_bound_h, self.bev_w, self.bev_h)
        self.bev_coords = self.bev_coords.reshape(-1, 4).permute(1, 0)

        self.global_map_raster_size = map_cfg['raster_size'] # TODO: check this
        self.global_map_dict = {}

        self.map_status = None
        self.epoch_point = -2
        self.update_value = 30
        self.use_mix = (self.load_map_path is not None or self.save_map_path is not None) and (dataset != 'av2')

    def load_map(self, device):
        self.global_map_dict = torch.load(self.load_map_path, map_location=device)

    def check_map(self, device, epoch, status):
        if status == 'train':
            self.fuse_method = 'all'   # To keep consistent with our initial setting
        else:
            self.fuse_method = self.fuse_method_val
        if self.map_status is None:
            self.epoch_point = epoch
            self.map_status = status
            if self.load_map_path is not None:
                self.load_map(device)
            else:
                self.create_map(device, status)
        elif status != self.map_status:
            self.epoch_point = epoch
            self.map_status = status
            self.create_map(device, status)
        elif epoch != self.epoch_point:
            self.epoch_point = epoch
            self.map_status = status
            self.reset_map()

    def reset_map(self):
        for city_name in self.city_list:
            self.global_map_dict[city_name].zero_()
            print("reset map", city_name, "for epoch", self.epoch_point, "status", self.map_status)
            if self.fuse_method == 'prob':
                self.global_map_dict[city_name] = self.global_map_dict[city_name] + 100

    def get_city_bound(self, city_name, status):
        if self.use_mix:
            return self.mix_min_lidar_loc[city_name], self.mix_max_lidar_loc[city_name]

        if status == 'train':
            return self.train_min_lidar_loc[city_name], self.train_max_lidar_loc[city_name]
        elif status == 'val':
            return self.val_min_lidar_loc[city_name], self.val_max_lidar_loc[city_name]
        elif status == 'mix':
            return self.mix_min_lidar_loc[city_name], self.mix_max_lidar_loc[city_name]

    def create_map(self, device, status):
        for city_name in self.city_list:
            city_min_bound, city_max_bound = self.get_city_bound(city_name, status)
            city_grid_size = (city_max_bound - city_min_bound) / np.array(self.global_map_raster_size, np.float32)
            map_height = city_grid_size[0]
            map_width = city_grid_size[1]
            map_height_ceil = int(np.ceil(map_height))
            map_width_ceil = int(np.ceil(map_width))
            city_map = torch.zeros((map_height_ceil, map_width_ceil, 3), dtype=self.map_type, device=device)
            if self.fuse_method == 'prob':
                city_map = city_map + 100
            print("create map", city_name, status, "on", device, "for epoch", self.epoch_point, "map: ", map_height_ceil, "*", map_width_ceil)
            self.global_map_dict[city_name] = city_map
        self.map_status = status

    def update_map(self, city_name, trans, raster, status):
        trans = self.bev_coords.new_tensor(trans)
        trans_bev_coords = trans @ self.bev_coords
        bev_coord_w = trans_bev_coords[0, :]
        bev_coord_h = trans_bev_coords[1, :]
        city_min_bound, city_max_bound = self.get_city_bound(city_name, status)

        bev_index_w = torch.floor((bev_coord_w - city_min_bound[0]) / self.bev_grid_len_w).to(torch.int64)
        bev_index_h = torch.floor((bev_coord_h - city_min_bound[1]) / self.bev_grid_len_w).to(torch.int64)
        bev_index_w = bev_index_w.reshape(self.bev_h, self.bev_w)
        bev_index_h = bev_index_h.reshape(self.bev_h, self.bev_w)
        bev_coord_mask = \
            (city_min_bound[0] <= bev_coord_w) & (bev_coord_w < city_max_bound[0]) & \
            (city_min_bound[1] <= bev_coord_h) & (bev_coord_h < city_max_bound[1])
        bev_coord_mask = bev_coord_mask.reshape(self.bev_h, self.bev_w)
        index_h, index_w = torch.where(bev_coord_mask)
        new_map = raster[index_h, index_w, :]
        old_map = self.global_map_dict[city_name][bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :]
        update_map = self.fuse_map(new_map, old_map)
        self.global_map_dict[city_name][bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] = update_map

    def get_map(self, city_name, trans, status):
        trans = self.bev_coords.new_tensor(trans)
        trans_bev_coords = trans @ self.bev_coords
        bev_coord_w = trans_bev_coords[0, :]
        bev_coord_h = trans_bev_coords[1, :]
        city_min_bound, city_max_bound = self.get_city_bound(city_name, status)

        bev_index_w = torch.floor((bev_coord_w - city_min_bound[0]) / self.bev_grid_len_w).to(torch.int64)
        bev_index_h = torch.floor((bev_coord_h - city_min_bound[1]) / self.bev_grid_len_h).to(torch.int64)
        bev_index_w = bev_index_w.reshape(self.bev_h, self.bev_w)
        bev_index_h = bev_index_h.reshape(self.bev_h, self.bev_w)
        bev_coord_mask = \
            (city_min_bound[0] <= bev_coord_w) & (bev_coord_w < city_max_bound[0]) & \
            (city_min_bound[1] <= bev_coord_h) & (bev_coord_h < city_max_bound[1])
        bev_coord_mask = bev_coord_mask.reshape(self.bev_h, self.bev_w)
        index_h, index_w = torch.where(bev_coord_mask)
        local_map = self.global_map_dict[city_name][bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :]
        if self.map_type == torch.uint8:
            local_map_float = local_map.float()
            if self.fuse_method == 'prob':
                local_map_float = (local_map_float - 100.0) / (self.update_value)
                local_map_float = torch.clamp(local_map_float, 0.0, 1.0)
            else:
                local_map_float = local_map_float / 255.0
            return local_map_float
        else:
            return local_map

    def fuse_map(self, new_map, old_map):
        if self.fuse_method == 'all':
            if self.map_type == torch.uint8:
                new_map = (new_map * 255).to(torch.uint8)
            return torch.max(new_map, old_map)
        elif self.fuse_method == 'prob':
            new_map = new_map * self.update_value + 100
            update_map = torch.max(new_map, old_map)
            new_map[new_map > 100] = 0
            new_map[new_map > 1] = -1
            update_map = update_map + new_map
            update_map = torch.clamp(update_map, 2, 240)
            update_map = update_map.to(self.map_type)
            return update_map

    def get_global_map(self):
        return self.global_map_dict

    def save_global_map(self):
        if self.save_map_path is not None:
            torch.save(self.global_map_dict, self.save_map_path)
            print("Save constructed map at", self.save_map_path, "!!!!!")

@DETECTORS.register_module()
class MergedTopoNet(MVXTwoStageDetector):

    def __init__(self,
                 bev_constructor=None,
                 bbox_head=None,
                 lane_head=None,
                 video_test_mode=False,
                 global_map_cfg=None,
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

        # variables
        num_decoder_layers = len(self.bbox_head.transformer.decoder.layers)
        self.num_decoder_layers = num_decoder_layers
        if len(self.bbox_head.transformer.decoder.layers) != len(self.pts_bbox_head.transformer.decoder.layers):
            raise NotImplementedError('not implemented when two decoders have different numbers of layers')
        embed_dims = self.bbox_head.transformer.embed_dims
        self.head_dim = int(embed_dims /
                            bbox_head.transformer.decoder.transformerlayers.attn_cfgs[
                                0].num_heads)
        self.num_object_queries = self.bbox_head.num_query + self.pts_bbox_head.num_query
        self.nq_te = self.bbox_head.num_query
        self.nq_cl = self.pts_bbox_head.num_query

        # projection layers
        self.proj_q_te = nn.ModuleList(
            [
                nn.Linear(embed_dims, embed_dims)
                for i in range(num_decoder_layers)
            ]
        )

        self.proj_k_te = nn.ModuleList(
            [
                nn.Linear(embed_dims, embed_dims)
                for i in range(num_decoder_layers)
            ]
        )
        self.proj_q_cl = nn.ModuleList(
            [
                nn.Linear(embed_dims, embed_dims)
                for i in range(num_decoder_layers)
            ]
        )

        self.proj_k_cl = nn.ModuleList(
            [
                nn.Linear(embed_dims, embed_dims)
                for i in range(num_decoder_layers)
            ]
        )

        # final projection layers
        # self.final_sub_proj = nn.Linear(embed_dims, embed_dims)
        # self.final_obj_proj = nn.Linear(embed_dims, embed_dims)
        # clcl
        self.final_sub_proj_clcl = nn.Linear(embed_dims, embed_dims)
        self.final_obj_proj_clcl = nn.Linear(embed_dims, embed_dims)
        # tecl
        self.final_sub_proj_tecl = nn.Linear(embed_dims, embed_dims)
        self.final_obj_proj_tecl = nn.Linear(embed_dims, embed_dims)

        # relation predictor gate
        self.rel_predictor_gate_tecl = nn.Linear(2 * embed_dims, 1)
        self.rel_predictor_gate_clcl = nn.Linear(2 * embed_dims, 1)

        # connectivity layers
        self.connectivity_layer_tecl = DeformableDetrMLPPredictionHead(
            input_dim=2*embed_dims,
            hidden_dim=embed_dims,
            output_dim=1,
            num_layers=3,
        )
        self.connectivity_layer_clcl = DeformableDetrMLPPredictionHead(
            input_dim=2 * embed_dims,
            hidden_dim=embed_dims,
            output_dim=1,
            num_layers=3,
        )

        # TODO: map
        if global_map_cfg is not None:
            self.global_map = GlobalMap(global_map_cfg)
            self.update_map = global_map_cfg['update_map']
        else:
            self.global_map = None
            self.update_map = False
        self.epoch = -1

    def update_global_map(self, img_metas, raster, status):
        bs = raster.shape[0]
        for i in range(bs):
            metas = img_metas[i]
            city_name = metas['map_location']
            trans = metas['lidar2global']
            self.global_map.update_map(city_name, trans, raster[i], status)

    def obtain_global_map(self, img_metas, status):
        bs = len(img_metas)
        bev_maps = []
        for i in range(bs):
            metas = img_metas[i]
            city_name = metas['map_location']
            trans = metas['lidar2global']
            local_map = self.global_map.get_map(city_name, trans, status)
            bev_maps.append(local_map)
        bev_maps = torch.stack(bev_maps)
        bev_maps = bev_maps.permute(1, 0, 2)
        return bev_maps

    def return_map(self):
        if self.update_map:
            self.global_map.save_global_map()
        # return self.global_map.get_global_map()

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

    def predict(self, img_feats, img_metas, prev_bev, front_view_img_feats, bbox_img_metas):

        # 1. prepare inputs
        te_feats = None
        te_cls_scores = None

        if self.global_map is not None:
            self.global_map.check_map(img_feats[0].device, self.epoch, 'train')
            local_map = self.obtain_global_map(img_metas, 'train')  # TODO
        else:
            local_map = None

        # TODO: figure out why the first dimension of local_map could be 0
        if local_map.shape[0] == 0:
            print('The 0th dimension of local_map is 0. Setting local_map to None')
            local_map = None

        # TODO: flip local map
        if local_map is not None:
            bs = local_map.shape[1]
            local_map = torch.flip(local_map.view(self.bev_constructor.bev_h, self.bev_constructor.bev_w, bs, 3), dims=[0]).view(-1, 1, 3)
        ########################################################################
        # import matplotlib.pyplot as plt
        # plt.imshow(local_map.squeeze(1).view(100, 200, 3).cpu())
        # plt.show()
        ########################################################################

        bev_feats, bev_pos = self.bev_constructor(img_feats, img_metas, prev_bev, local_map=local_map)

        # 2. go through the first half
        outputs_te_head_first_half = self.bbox_head.forward_first_half(front_view_img_feats, bbox_img_metas)
        outputs_te_transformer_first_half = self.bbox_head.transformer.forward_first_half(*outputs_te_head_first_half)
        outputs_cl_head_first_half = self.pts_bbox_head.forward_first_half(img_feats, bev_feats, img_metas, te_feats, te_cls_scores, local_map, bev_pos)
        outputs_cl_transformer_first_half = self.pts_bbox_head.transformer.forward_first_half(
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
        value_cl = outputs_cl_transformer_first_half['value'] # This is bev_embed

        # TODO: should this be None when local map is not None? (because of query initialization)
        if local_map is not None:
            query_pos_cl = None
        else:
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
                reference_points_input_te = reference_points_te[:, :, None] * torch.cat([valid_ratios_te, valid_ratios_te], -1)[:, None]
            else:
                assert reference_points_te.shape[-1] == 2
                reference_points_input_te = reference_points_te[:, :, None] * valid_ratios_te[:, None]

            # 0. input preparation (cl)
            reference_points_input_cl = reference_points_cl[..., :2].unsqueeze(2)

            # 1. self-attention after concatenating queries
            # te
            query_te, _, decoder_self_attention_q_te, _, decoder_self_attention_k_te, _ = \
                self.bbox_head.transformer.decoder.layers[
                    lid].forward_self_attention(query_te, None,
                                                query_pos_te=query_pos_te,
                                                query_pos_cl=None)
            # cl
            _, query_cl, _, decoder_self_attention_q_cl, _, decoder_self_attention_k_cl = \
                self.pts_bbox_head.transformer.decoder.layers[
                    lid].forward_self_attention(None, query_cl,
                                                query_pos_te=None,
                                                query_pos_cl=query_pos_cl)

            # Store q and k of te and cl in different lists
            decoder_attention_queries_te.append(decoder_self_attention_q_te)
            decoder_attention_keys_te.append(decoder_self_attention_k_te)
            decoder_attention_queries_cl.append(decoder_self_attention_q_cl)
            decoder_attention_keys_cl.append(decoder_self_attention_k_cl)


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
                intermediate_reference_points_cl.append(reference_points_cl)  # the each for each layer. check it. this is correct.

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

        # Relation prediction using Q and K
        bsz = reference_points_te.shape[0]
        num_object_queries = self.num_object_queries
        unscaling = self.head_dim ** 0.5

        # Unscaling & stacking attention queries
        projected_q_te = []
        for q, proj_q in zip(decoder_attention_queries_te, self.proj_q_te):
            projected_q_te.append(proj_q(q.permute(1, 0, 2) * unscaling))
        decoder_attention_queries_te = torch.stack(projected_q_te, -2)  # [bsz, num_object_queries, num_layers, d_model]
        del projected_q_te

        # Stacking attention keys
        projected_k_te = []
        for k, proj_k in zip(decoder_attention_keys_te, self.proj_k_te):
            projected_k_te.append(proj_k(k.permute(1, 0, 2)))
        decoder_attention_keys_te = torch.stack(projected_k_te, -2)  # [bsz, num_object_queries, num_layers, d_model]
        del projected_k_te

        projected_q_cl = []
        for q, proj_q in zip(decoder_attention_queries_cl, self.proj_q_cl):
            projected_q_cl.append(proj_q(q.permute(1, 0, 2) * unscaling))
        decoder_attention_queries_cl = torch.stack(projected_q_cl, -2)  # [bsz, num_object_queries, num_layers, d_model]
        del projected_q_cl

        # Stacking attention keys
        projected_k_cl = []
        for k, proj_k in zip(decoder_attention_keys_cl, self.proj_k_cl):
            projected_k_cl.append(proj_k(k.permute(1, 0, 2)))
        decoder_attention_keys_cl = torch.stack(projected_k_cl, -2)  # [bsz, num_object_queries, num_layers, d_model]
        del projected_k_cl

        # concat before pairwise concat
        decoder_attention_queries = torch.cat([decoder_attention_queries_te, decoder_attention_queries_cl], dim=1)
        decoder_attention_keys = torch.cat([decoder_attention_keys_te, decoder_attention_keys_cl], dim=1)

        # Pairwise concatenation
        decoder_attention_queries = decoder_attention_queries.unsqueeze(
            2).repeat(
            1, 1, num_object_queries, 1, 1
        )
        decoder_attention_keys = decoder_attention_keys.unsqueeze(
            1).repeat(
            1, num_object_queries, 1, 1, 1
        )
        relation_source = torch.cat(
            [decoder_attention_queries, decoder_attention_keys],
            dim=-1
        )  # [bsz, num_object_queries, num_object_queries, num_layers, 2*d_model]
        del decoder_attention_queries, decoder_attention_keys

        # add final hidden represetations separatly for tecl and clcl.
        #  Specifically, clcl should only receive query_cl while tecl receives both
        # clcl
        sequence_output_clcl = query_cl.permute(1, 0,2)
        subject_output_clcl = (
            self.final_sub_proj_clcl(sequence_output_clcl)
            .unsqueeze(2)
            .repeat(1, 1, self.nq_cl, 1)
        )
        object_output_clcl = (
            self.final_obj_proj_clcl(sequence_output_clcl)
            .unsqueeze(1)
            .repeat(1, self.nq_cl, 1, 1)
        )
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
        subject_output_tecl = (
            self.final_sub_proj_tecl(sequence_output_tecl)
            .unsqueeze(2)
            .repeat(1, 1, num_object_queries, 1)
        )
        object_output_tecl = (
            self.final_obj_proj_tecl(sequence_output_tecl)
            .unsqueeze(1)
            .repeat(1, num_object_queries, 1, 1)
        )
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
        rel_gate_tecl = torch.sigmoid(self.rel_predictor_gate_tecl(relation_source_tecl))
        gated_relation_source_tecl = torch.mul(rel_gate_tecl, relation_source_tecl).sum(dim=-2)

        rel_gate_clcl = torch.sigmoid(self.rel_predictor_gate_clcl(relation_source_clcl))
        gated_relation_source_clcl = torch.mul(rel_gate_clcl, relation_source_clcl).sum(dim=-2)

        # Connectivity
        pred_connectivity_tecl = self.connectivity_layer_tecl(gated_relation_source_tecl)
        pred_connectivity_clcl = self.connectivity_layer_clcl(gated_relation_source_clcl)
        del relation_source_tecl
        del gated_relation_source_clcl
        del rel_gate_tecl
        del rel_gate_clcl
        del relation_source

        # 4. go through the second half
        outputs_te_transformer_second_half = self.bbox_head.transformer.forward_second_half(*outputs_te_decoder, outputs_te_transformer_first_half['init_reference_out'])
        bbox_outs = self.bbox_head.forward_second_half(*outputs_te_transformer_second_half)
        outputs_cl_transformer_last_half = self.pts_bbox_head.transformer.forward_second_half(*outputs_cl_decoder, outputs_cl_transformer_first_half['init_reference_out'])
        outs = self.pts_bbox_head.forward_second_half(outputs_cl_transformer_last_half)

        outs['all_lclc_preds'] = [pred_connectivity_clcl]
        outs['all_lcte_preds'] = [pred_connectivity_tecl]

        return bbox_outs, outs, bev_feats

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

        # 2. Generate predictions
        bbox_outs, outs, _ = self.predict(img_feats, img_metas, prev_bev, front_view_img_feats, bbox_img_metas)

        # 3. Compute Losses
        te_losses = {}
        bbox_losses, te_assign_result = self.bbox_head.loss(bbox_outs, gt_bboxes, gt_labels, bbox_img_metas, gt_bboxes_ignore)
        for loss in bbox_losses:
            te_losses['bbox_head.' + loss] = bbox_losses[loss]
        num_gt_bboxes = sum([len(gt) for gt in gt_labels])
        if num_gt_bboxes == 0:
            for loss in te_losses:
                te_losses[loss] *= 0

        losses = dict()
        loss_inputs = [outs, gt_lanes_3d, gt_lane_labels_3d, gt_lane_adj, gt_lane_lcte_adj, te_assign_result]

        lane_losses = self.pts_bbox_head.my_loss(*loss_inputs, img_metas=img_metas, pred_connectivity_tecl=outs['all_lcte_preds'][-1], pred_connectivity_clcl=outs['all_lclc_preds'][-1])

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
        bbox_outs, outs, bev_feats = self.predict(x, img_metas, prev_bev, front_view_img_feats, bbox_img_metas)

        # 3. Get boxes, lanes, and relations
        bbox_results = self.bbox_head.get_bboxes(bbox_outs, bbox_img_metas, rescale=rescale)
        lane_results, lclc_results, lcte_results = self.pts_bbox_head.get_lanes(outs, img_metas, rescale=rescale)

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
