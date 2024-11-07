#!pip install av2==0.2.1
#!pip install numba==0.48.0

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import torch

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString

from tqdm import tqdm


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

        self.global_map_raster_size = map_cfg['raster_size']
        self.global_map_dict = {}

        self.map_status = None
        self.epoch_point = -2
        self.update_value = 30
        self.use_mix = self.load_map_path is not None or self.save_map_path is not None

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


import torch.nn as nn
from torch.autograd import Function
import native_rasterizer

MODE_BOUNDARY = "boundary"
MODE_MASK = "mask"
MODE_HARD_MASK = "hard_mask"

MODE_MAPPING = {
    MODE_BOUNDARY: 0,
    MODE_MASK: 1,
    MODE_HARD_MASK: 2
}


class SoftPolygonFunction(Function):
    @staticmethod
    def forward(ctx, vertices, width, height, inv_smoothness=1.0,
                mode=MODE_BOUNDARY, rasterized_dtype=None):
        ctx.width = width
        ctx.height = height
        ctx.inv_smoothness = inv_smoothness
        ctx.mode = MODE_MAPPING[mode]

        vertices = vertices.clone()
        ctx.device = vertices.device
        ctx.batch_size, ctx.number_vertices = vertices.shape[:2]

        if rasterized_dtype is None:
            rasterized = torch.FloatTensor(ctx.batch_size, ctx.height,
                                           ctx.width).fill_(0.0).to(
                device=ctx.device)
        elif rasterized_dtype == 'double':
            rasterized = torch.DoubleTensor(ctx.batch_size, ctx.height,
                                            ctx.width).fill_(0.0).to(
                device=ctx.device)

        contribution_map = torch.IntTensor(
            ctx.batch_size,
            ctx.height,
            ctx.width).fill_(0).to(device=ctx.device)
        rasterized, contribution_map = native_rasterizer.forward_rasterize(
            vertices, rasterized, contribution_map, width, height,
            inv_smoothness, ctx.mode)
        ctx.save_for_backward(vertices, rasterized, contribution_map)

        return rasterized  # , contribution_map

    @staticmethod
    def backward(ctx, grad_output):
        vertices, rasterized, contribution_map = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        # grad_vertices = torch.FloatTensor(
        #    ctx.batch_size, ctx.height, ctx.width, ctx.number_vertices, 2).fill_(0.0).to(device=ctx.device)
        grad_vertices = torch.FloatTensor(
            ctx.batch_size, ctx.number_vertices, 2).fill_(0.0).to(
            device=ctx.device)
        grad_vertices = native_rasterizer.backward_rasterize(
            vertices, rasterized, contribution_map, grad_output, grad_vertices,
            ctx.width, ctx.height, ctx.inv_smoothness, ctx.mode)

        return grad_vertices, None, None, None, None


class SoftPolygon(nn.Module):
    MODES = [MODE_BOUNDARY, MODE_MASK, MODE_HARD_MASK]

    def __init__(self, inv_smoothness=1.0, mode=MODE_BOUNDARY):
        super(SoftPolygon, self).__init__()

        self.inv_smoothness = inv_smoothness

        if not (mode in SoftPolygon.MODES):
            raise ValueError("invalid mode: {0}".format(mode))

        self.mode = mode

    def forward(self, vertices, width, height, p, color=False,
                rasterized_dtype=None):
        return SoftPolygonFunction.apply(vertices, width, height,
                                         self.inv_smoothness, self.mode,
                                         rasterized_dtype)


def pnp(vertices, width, height):
    device = vertices.device
    batch_size = vertices.size(0)
    polygon_dimension = vertices.size(1)

    y_index = torch.arange(0, height).to(device)
    x_index = torch.arange(0, width).to(device)

    grid_y, grid_x = torch.meshgrid(y_index, x_index)
    xp = grid_x.unsqueeze(0).repeat(batch_size, 1, 1).float()
    yp = grid_y.unsqueeze(0).repeat(batch_size, 1, 1).float()

    result = torch.zeros((batch_size, height, width)).bool().to(device)

    j = polygon_dimension - 1
    for vn in range(polygon_dimension):
        from_x = vertices[:, vn, 0].unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                                       height,
                                                                       width)
        from_y = vertices[:, vn, 1].unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                                       height,
                                                                       width)

        to_x = vertices[:, j, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, height,
                                                                    width)
        to_y = vertices[:, j, 1].unsqueeze(-1).unsqueeze(-1).repeat(1, height,
                                                                    width)

        has_condition = torch.logical_and((from_y > yp) != (to_y > yp), xp < (
                    (to_x - from_x) * (yp - from_y) / (to_y - from_y) + from_x))

        if has_condition.any():
            result[has_condition] = ~result[has_condition]

        j = vn

    signed_result = -torch.ones((batch_size, height, width), device=device)
    signed_result[result] = 1.0

    return signed_result


# used for verification purposes only.
class SoftPolygonPyTorch(nn.Module):
    def __init__(self, inv_smoothness=1.0):
        super(SoftPolygonPyTorch, self).__init__()

        self.inv_smoothness = inv_smoothness

    # vertices is N x P x 2
    def forward(self, vertices, width, height, p, color=False):
        device = vertices.device
        batch_size = vertices.size(0)
        polygon_dimension = vertices.size(1)

        inside_outside = pnp(vertices, width, height)

        # discrete points we will sample from.
        y_index = torch.arange(0, height).to(device)
        x_index = torch.arange(0, width).to(device)

        grid_y, grid_x = torch.meshgrid(y_index, x_index)
        grid_x = grid_x.unsqueeze(0).repeat(batch_size, 1, 1).float()
        grid_y = grid_y.unsqueeze(0).repeat(batch_size, 1, 1).float()

        # do this "per dimension"
        distance_segments = []
        over_segments = []
        color_segments = []
        for from_index in range(polygon_dimension):
            segment_result = torch.zeros((batch_size, height, width)).to(device)
            from_vertex = vertices[:, from_index].unsqueeze(-1).unsqueeze(-1)

            if from_index == (polygon_dimension - 1):
                to_vertex = vertices[:, 0].unsqueeze(-1).unsqueeze(-1)
            else:
                to_vertex = vertices[:, from_index + 1].unsqueeze(-1).unsqueeze(
                    -1)

            x2_sub_x1 = to_vertex[:, 0] - from_vertex[:, 0]
            y2_sub_y1 = to_vertex[:, 1] - from_vertex[:, 1]
            square_segment_length = x2_sub_x1 * x2_sub_x1 + y2_sub_y1 * y2_sub_y1 + 0.00001

            # figure out if this is a major/minor segment (todo?)
            x_sub_x1 = grid_x - from_vertex[:, 0]
            y_sub_y1 = grid_y - from_vertex[:, 1]
            x_sub_x2 = grid_x - to_vertex[:, 0]
            y_sub_y2 = grid_y - to_vertex[:, 1]

            # dot between the given point and first vertex and first vertex and second vertex.
            dot = ((x_sub_x1 * x2_sub_x1) + (
                        y_sub_y1 * y2_sub_y1)) / square_segment_length

            # needlessly computed sometimes.
            x_proj = grid_x - (from_vertex[:, 0] + dot * x2_sub_x1)
            y_proj = grid_y - (from_vertex[:, 1] + dot * y2_sub_y1)

            from_closest = dot < 0
            to_closest = dot > 1
            interior_closest = (dot >= 0) & (dot <= 1)

            segment_result[from_closest] = x_sub_x1[from_closest] ** 2 + \
                                           y_sub_y1[from_closest] ** 2
            segment_result[to_closest] = x_sub_x2[to_closest] ** 2 + y_sub_y2[
                to_closest] ** 2
            segment_result[interior_closest] = x_proj[interior_closest] ** 2 + \
                                               y_proj[interior_closest] ** 2

            distance_map = -segment_result
            distance_segments.append(distance_map)

            signed_map = torch.sigmoid(
                -distance_map * inside_outside / self.inv_smoothness)
            over_segments.append(signed_map)

        F_max, F_arg = torch.max(torch.stack(distance_segments, dim=-1), dim=-1)
        F_theta = torch.gather(torch.stack(over_segments, dim=-1), dim=-1,
                               index=F_arg.unsqueeze(-1))[..., 0]

        return F_theta


def decode_raster_single(traj,
                         bev_h=100,
                         bev_w=200,
                         inv_smoothness=2.0,
                         use_dilate=False):
    '''
    pts: (N, Num_p, 2)
    labels: (N, 1)
    '''
    pts = traj['pts'].double().cuda()
    labels = traj['labels'].cuda()
    height = bev_h
    width = bev_w

    new_pts = pts.clone().cuda()
    new_pts[..., 0:1] = pts[..., 0:1] * width
    # new_pts[..., 1:2] = (1.0 - pts[..., 1:2]) * height
    new_pts[..., 1:2] = (pts[..., 1:2]) * height

    # if DEBUG:
    #     for vector in new_pts:
    #         plt.plot(vector[:, 0].cpu().numpy(), vector[:, 1].cpu().numpy())
    #         plt.tight_layout()
    #         plt.savefig(f'vis_pts.png')

    divider_index = torch.nonzero(labels == 0, as_tuple=True)
    ped_crossing_index = torch.nonzero(labels == 1, as_tuple=True)
    boundary_index = torch.nonzero(labels == 2, as_tuple=True)
    divider_pts = new_pts[divider_index]
    ped_crossing_pts = new_pts[ped_crossing_index]
    boundary_pts = new_pts[boundary_index]
    rasterized_results = torch.zeros(3, height, width, device=pts.device)
    if divider_pts.shape[0] > 0:
        HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary",
                                           inv_smoothness=inv_smoothness)
        rasterized_line = HARD_CUDA_RASTERIZER(divider_pts, int(width),
                                               int(height), 1.0,
                                               rasterized_dtype='double')
        rasterized_line, _ = torch.max(rasterized_line, 0)
        rasterized_results[0] = rasterized_line

    if ped_crossing_pts.shape[0] > 0:
        HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask",
                                           inv_smoothness=inv_smoothness)
        rasterized_poly = HARD_CUDA_RASTERIZER(ped_crossing_pts, int(width),
                                               int(height), 1.0,
                                               rasterized_dtype='double')
        rasterized_poly, _ = torch.max(rasterized_poly, 0)
        rasterized_results[1] = rasterized_poly

    if boundary_pts.shape[0] > 0:
        HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary",
                                           inv_smoothness=inv_smoothness)
        rasterized_line = HARD_CUDA_RASTERIZER(boundary_pts, int(width),
                                               int(height), 1.0,
                                               rasterized_dtype='double')
        rasterized_line, _ = torch.max(rasterized_line, 0)
        rasterized_results[2] = rasterized_line

    if use_dilate:
        max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        rasterized_results = max_pool(rasterized_results)

    return rasterized_results.cpu()


def decode_raster(trajs):
    batch_size = len(trajs)
    predictions_list = []
    for i in range(batch_size):
        predictions_list.append(decode_raster_single(trajs[i]))
    return predictions_list


def get_traj_mask(trajs):
    with torch.no_grad():
        raster_lists = decode_raster(trajs)
        raster_tensor = torch.stack(raster_lists, dim=0)
        # bs, n, bev_h, bev_w = raster_tensor.shape
        raster_tensor = raster_tensor.permute(0, 2, 3, 1)  # -> b, h, w, n
        return raster_tensor


def get_vectors(trajs, labels, patch_box, patch_angle, patch,
                fixed_num=15):
    patch_x = patch_box[0]
    patch_y = patch_box[1]

    line_list = []
    selected_labels = []
    for idx, traj in enumerate(trajs):
        if traj.is_empty:
            continue
        new_line = traj.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.rotate(new_line, -patch_angle,
                                       origin=(patch_x, patch_y),
                                       use_radians=False)
            new_line = affinity.affine_transform(new_line,
                                                 [1.0, 0.0, 0.0, 1.0, -patch_x,
                                                  -patch_y])
            line_list.append(new_line)
            selected_labels.append(labels[idx])
    vectors = []
    for line in line_list:
        distances = np.linspace(0, line.length, fixed_num)
        sampled_points = np.array(
            [list(line.interpolate(distance).coords) for distance in
             distances]).reshape(-1, 2)
        vectors.append(sampled_points)
    vectors = np.array(vectors)
    selected_labels = np.array(selected_labels)
    return vectors, selected_labels


# Start Creating Global Map
data_root = Path('/data_storage/liyang/argoverse2/sensor/train')
# data_root = Path('/DATA_EDS2/zhangzz2401/zhangzz2401/MapTR-maptrv2/data/argoverse2/sensor/train')
# data_root = Path('/data21/2024/liy2408/ram_disk/mini_train')
loader = AV2SensorDataLoader(data_root, data_root)
list_log_id = os.listdir(data_root)

scenes_trajs_WDC = dict()
scenes_trajs_MIA = dict()
scenes_trajs_PAO = dict()
scenes_trajs_PIT = dict()
scenes_trajs_ATX = dict()
scenes_trajs_DTW = dict()

scenes_trajs_WDC_other = dict()
scenes_trajs_MIA_other = dict()
scenes_trajs_PAO_other = dict()
scenes_trajs_PIT_other = dict()
scenes_trajs_ATX_other = dict()
scenes_trajs_DTW_other = dict()

print('Getting trajectories:')
for log_id in tqdm(list_log_id, desc='Getting trajectories'):

    city_name = loader.get_city_name(log_id)
    all_time_stamps = loader.get_ordered_log_lidar_timestamps(log_id)

    # ego
    ego_trajectory = []
    EPS = 0.01
    SAMPLE_INTERVAL = 1
    list_trajectory_ego_objects = []
    for lidar_timestamp_ns in all_time_stamps[::SAMPLE_INTERVAL]:
        city_SE3_ego = loader.get_city_SE3_ego(log_id=log_id,
                                               timestamp_ns=lidar_timestamp_ns)
        T_ego_to_global = np.eye(4)
        T_ego_to_global[:3, :3] = city_SE3_ego.rotation
        T_ego_to_global[:3, -1] = city_SE3_ego.translation

        T_agent_to_global = T_ego_to_global

        global_xy = T_agent_to_global[:2, -1]

        category = 'ego'

        result_dict = {
            'traj': LineString(np.array([global_xy, global_xy + EPS])),
            'label': category,
            'T_agent_to_global': T_agent_to_global,
            }
        list_trajectory_ego_objects.append(result_dict)

    if city_name == 'WDC':
        scenes_trajs_WDC[log_id] = list_trajectory_ego_objects
    elif city_name == 'MIA':
        scenes_trajs_MIA[log_id] = list_trajectory_ego_objects
    elif city_name == 'PAO':
        scenes_trajs_PAO[log_id] = list_trajectory_ego_objects
    elif city_name == 'PIT':
        scenes_trajs_PIT[log_id] = list_trajectory_ego_objects
    elif city_name == 'ATX':
        scenes_trajs_ATX[log_id] = list_trajectory_ego_objects
    elif city_name == 'DTW':
        scenes_trajs_DTW[log_id] = list_trajectory_ego_objects
    else:
        raise ValueError(city_name + 'is not a valid city name')

    # Trajectory from other vehicles
    EPS = 0.01
    SAMPLE_INTERVAL = 1
    list_trajectory_other_objects = []
    for lidar_timestamp_ns in all_time_stamps[::SAMPLE_INTERVAL]:

        city_SE3_ego = loader.get_city_SE3_ego(log_id=log_id,
                                               timestamp_ns=lidar_timestamp_ns)
        T_ego_to_global = np.eye(4)
        T_ego_to_global[:3, :3] = city_SE3_ego.rotation
        T_ego_to_global[:3, -1] = city_SE3_ego.translation

        labels_at_lidar_timestamp = loader.get_labels_at_lidar_timestamp(log_id,
                                                                         lidar_timestamp_ns=lidar_timestamp_ns)

        # other objects
        for i in range(len(labels_at_lidar_timestamp)):
            currenet_labels_at_lidar_timestamp = labels_at_lidar_timestamp[i]

            R_agent_to_ego = currenet_labels_at_lidar_timestamp.dst_SE3_object.rotation
            t_agent_to_ego = currenet_labels_at_lidar_timestamp.dst_SE3_object.translation

            T_agent_to_ego = np.eye(4)
            T_agent_to_ego[:3, :3] = R_agent_to_ego
            T_agent_to_ego[:3, -1] = t_agent_to_ego
            T_agent_to_global = T_ego_to_global @ T_agent_to_ego

            global_xy = T_agent_to_global[:2, -1]

            category = currenet_labels_at_lidar_timestamp.category

            result_dict = {
                'traj': LineString(np.array([global_xy, global_xy + EPS])),
                'label': category,
                # 'T_agent_to_ego': T_agent_to_ego,
                # 'T_ego_to_global': T_ego_to_global,
                'T_agent_to_global': T_agent_to_global
                }
            list_trajectory_other_objects.append(result_dict)

        if city_name == 'WDC':
            scenes_trajs_WDC_other[log_id] = list_trajectory_other_objects
        elif city_name == 'MIA':
            scenes_trajs_MIA_other[log_id] = list_trajectory_other_objects
        elif city_name == 'PAO':
            scenes_trajs_PAO_other[log_id] = list_trajectory_other_objects
        elif city_name == 'PIT':
            scenes_trajs_PIT_other[log_id] = list_trajectory_other_objects
        elif city_name == 'ATX':
            scenes_trajs_ATX_other[log_id] = list_trajectory_other_objects
        elif city_name == 'DTW':
            scenes_trajs_DTW_other[log_id] = list_trajectory_other_objects
        else:
            raise ValueError(city_name + 'is not a valid city name')


print("Number of ego trajectories in each city:")
print('WDC:', len(scenes_trajs_WDC))
print('MIA:', len(scenes_trajs_MIA))
print('PAO:', len(scenes_trajs_PAO))
print('PIT:', len(scenes_trajs_PIT))
print('ATX:', len(scenes_trajs_ATX))
print('DTW:', len(scenes_trajs_DTW))
total_number_of_trajectories = 0
total_number_of_trajectories += len(scenes_trajs_WDC)
total_number_of_trajectories += len(scenes_trajs_MIA)
total_number_of_trajectories += len(scenes_trajs_PAO)
total_number_of_trajectories += len(scenes_trajs_PIT)
total_number_of_trajectories += len(scenes_trajs_ATX)
total_number_of_trajectories += len(scenes_trajs_DTW)
print("Total number of ego trajectories:", total_number_of_trajectories)
print()
print()
print()


print("Number of other trajectories in each city:")
print('WDC:', len(scenes_trajs_WDC_other))
print('MIA:', len(scenes_trajs_MIA_other))
print('PAO:', len(scenes_trajs_PAO_other))
print('PIT:', len(scenes_trajs_PIT_other))
print('ATX:', len(scenes_trajs_ATX_other))
print('DTW:', len(scenes_trajs_DTW_other))
total_number_of_trajectories = 0
total_number_of_trajectories += len(scenes_trajs_WDC_other)
total_number_of_trajectories += len(scenes_trajs_MIA_other)
total_number_of_trajectories += len(scenes_trajs_PAO_other)
total_number_of_trajectories += len(scenes_trajs_PIT_other)
total_number_of_trajectories += len(scenes_trajs_ATX_other)
total_number_of_trajectories += len(scenes_trajs_DTW_other)
print("Total number of other trajectories:", total_number_of_trajectories)
print()
print()
print()

# def plot_city_trajectory_map_all(scenes_trajs, scenes_trajs_other,
#                                  city_name=None):
#     for scene_id in scenes_trajs:
#         list_traj_ego = scenes_trajs[scene_id]
#         for dict_ego in tqdm(list_traj_ego, desc=city_name):
#             x, y = dict_ego['traj'].xy
#
#             plt.scatter(x, y, s=1, c='r')
#
#     for scene_id in scenes_trajs_other:
#
#         list_traj_other = scenes_trajs_other[scene_id]
#
#         for dict_other in tqdm(list_traj_other, desc=city_name):
#             x, y = dict_other['traj'].xy
#
#             if dict_other['label'] == 'ego':
#                 plt.scatter(x, y, s=1, c='r')
#             else:
#                 plt.scatter(x, y, s=1, c='g')
#
#     if city_name is not None:
#         plt.title(city_name)
#     plt.show()
#     # plt.savefig('visualization_' + city_name + '.png')
#
#
# plot_city_trajectory_map_all(scenes_trajs_WDC, scenes_trajs_WDC_other,
#                              city_name='WDC')
# plot_city_trajectory_map_all(scenes_trajs_MIA, scenes_trajs_MIA_other,
#                              city_name='MIA')
# plot_city_trajectory_map_all(scenes_trajs_PAO, scenes_trajs_PAO_other,
#                              city_name='PAO')
# plot_city_trajectory_map_all(scenes_trajs_PIT, scenes_trajs_PIT_other,
#                              city_name='PIT')
# plot_city_trajectory_map_all(scenes_trajs_ATX, scenes_trajs_ATX_other,
#                              city_name='ATX')
# plot_city_trajectory_map_all(scenes_trajs_DTW, scenes_trajs_DTW_other,
#                              city_name='DTW')


# Initialize Global Map
# using parameters from ~/TopoNet/projects/configs/merged_subset_A.py

point_cloud_range = [-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]
bev_h_ = 100
bev_w_ = 200

map_cfg = dict(
    pc_range=point_cloud_range,
    bev_h=bev_h_,
    bev_w=bev_w_,
    fuse_method='prob',  # all or prob
    raster_size=[0.30, 0.30],
    dataset='av2',
    load_map_path=None,
    save_map_path='openlanev2_global_map_with_other_cars_no_subsampling.pt',
    update_map=False,
)


# Mappings
# https://argoverse.github.io/user-guide/datasets/sensor.html#annotations
category_mappings = {
    'REGULAR_VEHICLE': 1,
    'PEDESTRIAN': -1,
    'BOLLARD': -1,
    'CONSTRUCTION_CONE': -1,
    'CONSTRUCTION_BARREL': -1,
    'STOP_SIGN': -1,
    'BICYCLE': -1,
    'LARGE_VEHICLE': 2,
    'WHEELED_DEVICE': -1,
    'BUS': 2,
    'BOX_TRUCK': 2,
    'SIGN': -1,
    'TRUCK': 2,
    'MOTORCYCLE': 2,
    'BICYCLIST': -1,
    'VEHICULAR_TRAILER': -1,
    'TRUCK_CAB': 2,
    'MOTORCYCLIST': 2,
    'DOG': -1,
    'SCHOOL_BUS': 2,
    'WHEELED_RIDER': -1,
    'STROLLER': -1,
    'ARTICULATED_BUS': 2,
    'MESSAGE_BOARD_TRAILER': -1,
    'MOBILE_PEDESTRIAN_SIGN': -1,
    'WHEELCHAIR': -1,
    'RAILED_VEHICLE': -1,
    'OFFICIAL_SIGNALER': -1,
    'TRAFFIC_LIGHT_TRAILER': -1,
    'ANIMAL': -1,
    'MOBILE_PEDESTRIAN_CROSSING_SIGN': -1
}
#
# print("category: 1:")
# for k in category_mappings:
#     if category_mappings[k] == 1:
#         print(k)
# print()
#
# print("category: 2:")
# for k in category_mappings:
#     if category_mappings[k] == 2:
#         print(k)
# print()
#
# print("category: -1:")
# for k in category_mappings:
#     if category_mappings[k] == -1:
#         print(k)
# print()


# Update Global Map
global_map = GlobalMap(map_cfg)
global_map.use_mix = False

global_map.check_map('cpu', 0, 'train')

list_city_names = ['WDC', 'MIA', 'PAO', 'PIT', 'ATX', 'DTW']
list_scenes_trajs_dicts = [scenes_trajs_WDC,
                           scenes_trajs_MIA,
                           scenes_trajs_PAO,
                           scenes_trajs_PIT,
                           scenes_trajs_ATX,
                           scenes_trajs_DTW]
list_scenes_trajs_dicts_other = [scenes_trajs_WDC_other,
                                 scenes_trajs_MIA_other,
                                 scenes_trajs_PAO_other,
                                 scenes_trajs_PIT_other,
                                 scenes_trajs_ATX_other,
                                 scenes_trajs_DTW_other]

first_value = 60
second_value = 30

print()
print("Updating Global Map:")
for city_name, scenes_trajs_dict, scenes_trajs_dict_other in tqdm(zip(
        list_city_names, list_scenes_trajs_dicts,
        list_scenes_trajs_dicts_other), total=len(list_city_names), desc='Updating global map'):

    for log_id in scenes_trajs_dict.keys():

        # Ego
        traj_list_ego = scenes_trajs_dict[log_id]
        for dict_ego in traj_list_ego:
            T_agent_to_global = dict_ego['T_agent_to_global']
            traj_ego = dict_ego['traj']

            cur_trajs = [traj_ego]
            cur_labels = [0]

            # 1. patch_box
            patch_box = (
            T_agent_to_global[0, -1], T_agent_to_global[1, -1], first_value,
            second_value)
            lidar2global_rot = Quaternion._from_matrix(
                T_agent_to_global[:3, :3])  # <------
            rotation = Quaternion(lidar2global_rot)
            # 2. patch_angle
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            # 3. patch
            patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)

            vectors, labels = get_vectors(cur_trajs, cur_labels, patch_box,
                                          patch_angle, patch)

            if vectors.shape[0] == 0:
                print('Ego empty vector')
                continue

            vectors[:, :, 0] = (vectors[:, :, 0] + (
                        float(second_value) / 2)) / float(second_value)
            vectors[:, :, 1] = (vectors[:, :, 1] + (
                        float(first_value) / 2)) / float(first_value)

            trajs = []
            trajs.append(
                dict(pts=torch.tensor(vectors), labels=torch.tensor(labels), ))

            raster = get_traj_mask(trajs)
            # raise NotImplementedError()

            trans = np.eye(4)
            trans[:3, :3] = T_agent_to_global[:3, :3]  # <------
            trans[:3, 3] = T_agent_to_global[:3, -1]  # <------

            global_map.update_map(city_name, trans, raster[0].float(), 'train')

        # Other objects
        traj_list_other = scenes_trajs_dict_other[log_id]
        for dict_other in traj_list_other:
            # T_ego_to_global = dict_other['T_ego_to_global']
            # T_agent_to_ego = dict_other['T_agent_to_ego']
            traj_other = dict_other['traj']
            label_other = dict_other['label']
            T_agent_to_global = dict_other['T_agent_to_global']

            cur_trajs = [traj_other]

            try:
                mapped_label = category_mappings[label_other]
            except:
                category_mappings[label_other] = -1
                print()
                print("Adding missing key to dict:", label_other)
                print()
                mapped_label = category_mappings[label_other]

            if mapped_label == -1:
                continue
            cur_labels = [mapped_label]

            # 1. patch_box
            patch_box = (
            T_agent_to_global[0, -1], T_agent_to_global[1, -1], first_value,
            second_value)
            lidar2global_rot = Quaternion._from_matrix(
                T_agent_to_global[:3, :3])  # <------
            rotation = Quaternion(lidar2global_rot)
            # 2. patch_angle
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            # 3. patch
            patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)

            vectors, labels = get_vectors(cur_trajs, cur_labels, patch_box,
                                          patch_angle, patch)

            if vectors.shape[0] == 0:
                print('Other empty vector')
                continue

            vectors[:, :, 0] = (vectors[:, :, 0] + (
                        float(second_value) / 2)) / float(second_value)
            vectors[:, :, 1] = (vectors[:, :, 1] + (
                        float(first_value) / 2)) / float(first_value)

            trajs = []
            trajs.append(
                dict(pts=torch.tensor(vectors), labels=torch.tensor(labels), ))

            raster = get_traj_mask(trajs)

            trans = np.eye(4)
            trans[:3, :3] = T_agent_to_global[:3, :3]  # <------
            trans[:3, 3] = T_agent_to_global[:3, -1]  # <------

            global_map.update_map(city_name, trans, raster[0].float(), 'train')


# Save Global Map
print()
print('Saving map to:', global_map.save_map_path)
global_map.save_global_map()
print()
print()


# CV2 Visualize Global Map¶
import cv2

def vis_global_map(city_name):
    point_cloud_range = [-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]
    bev_h_ = 100
    bev_w_ = 200

    map_cfg = dict(
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        fuse_method='prob',  # all or prob
        raster_size=[0.30, 0.30],
        dataset='av2',
        load_map_path='openlanev2_global_map_with_other_cars_no_subsampling.pt',
        save_map_path=None,
        update_map=True,
    )

    global_map = GlobalMap(map_cfg)
    global_map.load_map('cpu')
    city_map = global_map.global_map_dict[city_name]
    city_map_np = city_map.numpy()
    out_path = 'output_image_' + city_name + '.png'
    cv2.imwrite(out_path, city_map_np)
    print('Visualization saved to:', out_path)

for city_name in tqdm(list_city_names, desc='Visualizing mmdet global map'):
    vis_global_map(city_name)
