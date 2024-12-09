#---------------------------------------------------------------------------------------#
# Graph-based Topology Reasoning for Driving Scenes (https://arxiv.org/abs/2304.05277)  #
# Source code: https://github.com/OpenDriveLab/TopoNet                                  #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import LoadAnnotations3D
import torch
import os
from mmdet3d.core.points import BasePoints, get_points_type
# from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams
from av2.utils.io import read_lidar_sweep
import pandas as pd

__all__ = ["load_augmented_point_cloud", "reduce_LiDAR_beams"]


def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    # NOTE: following Tianwei's implementation, it is hard coded for nuScenes
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    # NOTE: path definition different from Tianwei's implementation.
    tokens = path.split("/")
    vp_dir = "_VIRTUAL" if reduce_beams == 32 else f"_VIRTUAL_{reduce_beams}BEAMS"
    seg_path = os.path.join(
        *tokens[:-3],
        "virtual_points",
        tokens[-3],
        tokens[-2] + vp_dir,
        tokens[-1] + ".pkl.npy",
    )
    assert os.path.exists(seg_path)
    data_dict = np.load(seg_path, allow_pickle=True).item()

    virtual_points1 = data_dict["real_points"]
    # NOTE: add zero reflectance to virtual points instead of removing them from real points
    virtual_points2 = np.concatenate(
        [
            data_dict["virtual_points"][:, :3],
            np.zeros([data_dict["virtual_points"].shape[0], 1]),
            data_dict["virtual_points"][:, 3:],
        ],
        axis=-1,
    )

    points = np.concatenate(
        [
            points,
            np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]),
        ],
        axis=1,
    )
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1
    )
    # note: this part is different from Tianwei's implementation, we don't have duplicate foreground real points.
    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1
    if virtual:
        virtual_points2 = np.concatenate(
            [virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1
        )
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
    return points


def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    # print(pts.size())
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    phi = torch.atan2(pts[:, 1], pts[:, 0])

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275
    # beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
    #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]

    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    # [?] pick the 14th beam
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
            theta > (beam_range[chosen_beam_id] - 0.012)
        )
    else:
        raise NotImplementedError
    # points = copy.copy(pts)
    points = pts[mask]
    # print(points.size())
    return points.numpy()


@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFilesToponet(object):

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [_.astype(np.float32) for _ in img]
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = [img_.shape for img_ in img]
        results['ori_shape'] = [img_.shape for img_ in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [img_.shape for img_ in img]
        results['crop_shape'] = [np.zeros(2) for img_ in img]
        results['scale_factor'] = [1.0 for img_ in img]
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(object):

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [_.astype(np.float32) for _ in img]
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = [img_.shape for img_ in img]
        results['ori_shape'] = [img_.shape for img_ in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [img_.shape for img_ in img]
        results['crop_shape'] = [np.zeros(2) for img_ in img]
        results['scale_factor'] = [1.0 for img_ in img]
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3DLane(LoadAnnotations3D):
    """Load Annotations3D Lane.

    Args:
        with_lane_3d (bool, optional): Whether to load 3D Lanes.
            Defaults to True.
        with_lane_label_3d (bool, optional): Whether to load 3D Lanes Labels.
            Defaults to True.
        with_lane_adj (bool, optional): Whether to load Lane-Lane Adjacency.
            Defaults to True.
        with_lane_lcte_adj (bool, optional): Whether to load Lane-TE Adjacency.
            Defaults to False.
    """

    def __init__(self,
                 with_lane_3d=True,
                 with_lane_label_3d=True,
                 with_lane_adj=True,
                 with_lane_lcte_adj=False,
                 with_bbox_3d=False,
                 with_label_3d=False,
                 **kwargs):
        super().__init__(with_bbox_3d, with_label_3d, **kwargs)
        self.with_lane_3d = with_lane_3d
        self.with_lane_label_3d = with_lane_label_3d
        self.with_lane_adj = with_lane_adj
        self.with_lane_lcte_adj = with_lane_lcte_adj

    def _load_lanes_3d(self, results):
        results['gt_lanes_3d'] = results['ann_info']['gt_lanes_3d']
        if self.with_lane_label_3d:
            results['gt_lane_labels_3d'] = results['ann_info']['gt_lane_labels_3d']
        if self.with_lane_adj:
            results['gt_lane_adj'] = results['ann_info']['gt_lane_adj']
        if self.with_lane_lcte_adj:
            results['gt_lane_lcte_adj'] = results['ann_info']['gt_lane_lcte_adj']
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_lane_3d:
            results = self._load_lanes_3d(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = super().__repr__()
        repr_str += f'{indent_str}with_lane_3d={self.with_lane_3d}, '
        repr_str += f'{indent_str}with_lane_lable_3d={self.with_lane_lable_3d}, '
        repr_str += f'{indent_str}with_lane_adj={self.with_lane_adj}, '
        return repr_str


@PIPELINES.register_module()
class CustomLoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        # if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
        if True:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points

        ########################################################################
        if False:
            print()
            print()
            print(points.tensor[:, 0].min(), points.tensor[:, 0].max(), points.tensor[:, 0].mean())
            print(points.tensor[:, 1].min(), points.tensor[:, 1].max(), points.tensor[:, 1].mean())
            print(points.tensor[:, 2].min(), points.tensor[:, 2].max(), points.tensor[:, 2].mean())
            save_path = 'debug_points/' + 'multi_sweep_points.ply'
            with open(save_path, 'w') as file:
                file.write("ply\n")
                file.write("format ascii 1.0\n")
                file.write(f"element vertex {points.shape[0]}\n")
                file.write("property float x\n")
                file.write("property float y\n")
                file.write("property float z\n")
                file.write("end_header\n")
                for point in points.tensor:
                    file.write(f"{point[0]} {point[1]} {point[2]}\n")
        ########################################################################

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"



@PIPELINES.register_module()
class CustomLoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        str_lidar_path = str(lidar_path)
        lidar_time_stamp = int(str_lidar_path[str_lidar_path.rfind('/')+1:str_lidar_path.rfind('.feather')])
        # points = read_lidar_sweep(lidar_path)
        points = pd.read_feather(lidar_path).to_numpy()[:, self.use_dim]

        ########################################################################
        if False:
            print()
            print()
            print(points[:, 0].min(), points[:, 0].max(), points[:, 0].mean())
            print(points[:, 1].min(), points[:, 1].max(), points[:, 1].mean())
            print(points[:, 2].min(), points[:, 2].max(), points[:, 2].mean())
            save_path = 'debug_points/' + 'loaded_points.ply'
            with open(save_path, 'w') as file:
                file.write("ply\n")
                file.write("format ascii 1.0\n")
                file.write(f"element vertex {points.shape[0]}\n")
                file.write("property float x\n")
                file.write("property float y\n")
                file.write("property float z\n")
                file.write("end_header\n")
                for point in points:
                    file.write(f"{point[0]} {point[1]} {point[2]}\n")
        ########################################################################

        points = np.concatenate([points, np.zeros([points.shape[0], 2], dtype=points.dtype)], axis=1)

        # points = self._load_points(lidar_path)
        # points = points.reshape(-1, self.load_dim)
        # if self.reduce_beams and self.reduce_beams < 32:
        #     points = reduce_LiDAR_beams(points, self.reduce_beams)
        # points = points[:, self.use_dim]
        attribute_dims = None

        # if self.shift_height:
        #     floor_height = np.percentile(points[:, 2], 0.99)
        #     height = points[:, 2] - floor_height
        #     points = np.concatenate(
        #         [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
        #     )
        #     attribute_dims = dict(height=3)
        #
        # if self.use_color:
        #     assert len(self.use_dim) >= 6
        #     if attribute_dims is None:
        #         attribute_dims = dict()
        #     attribute_dims.update(
        #         dict(
        #             color=[
        #                 points.shape[1] - 3,
        #                 points.shape[1] - 2,
        #                 points.shape[1] - 1,
        #             ]
        #         )
        #     )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points
        results["timestamp"] = lidar_time_stamp


        return results


