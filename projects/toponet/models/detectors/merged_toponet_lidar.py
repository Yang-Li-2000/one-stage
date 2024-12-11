# ---------------------------------------------------------------------------------------#
# Graph-based Topology Reasoning for Driving Scenes (https://arxiv.org/abs/2304.05277)  #
# Source code: https://github.com/OpenDriveLab/TopoNet                                  #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
# ---------------------------------------------------------------------------------------#

import os
# Add the path to LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/data3/liyang/.conda/envs/merged_for_smerf/lib/python3.8/site-packages/torch/lib'
# Now, import sparse_conv_ext
from projects.sparse_conv_ext import sparse_conv_ext


import time
import copy
import numpy as np
import torch
from torch import nn

from mmcv.runner import force_fp32, auto_fp16

from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import builder
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet3d.models.builder import build_neck
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from ...utils.builder import build_bev_constructor
from mmdet.models.utils.transformer import inverse_sigmoid

from mmcv.ops import SparseConvTensor

from .model.deformable_detr import (
    DeformableDetrHungarianMatcher,
    DeformableDetrMLPPredictionHead,
    DeformableDetrModel,
    DeformableDetrPreTrainedModel,
    inverse_sigmoid,
)

# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
# from ..builder import MIDDLE_ENCODERS
from mmdet3d.models.builder import MIDDLE_ENCODERS

from collections import OrderedDict

from torch.nn.parameter import Parameter

import math
import numpy as np
import torch
from mmcv.cnn import CONV_LAYERS
from torch.nn import init

import sys

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from mmdet3d.ops.sparse_block import SparseModule


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[
            i
        ] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
        output_size.append(size)
    return output_size


def get_indice_pairs(
    indices,
    batch_size,
    spatial_shape,
    ksize=3,
    stride=1,
    padding=0,
    dilation=1,
    out_padding=0,
    subm=False,
    transpose=False,
    grid=None,
):
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(
                spatial_shape, ksize, stride, padding, dilation, out_padding
            )
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride, padding, dilation)

    else:
        out_shape = spatial_shape
    if grid is None:
        if ndim == 2:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_2d
        elif ndim == 3:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_3d
        elif ndim == 4:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_4d
        else:
            raise NotImplementedError
        return get_indice_pairs_func(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )
    else:
        if ndim == 2:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_grid_2d
        elif ndim == 3:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_grid_3d
        else:
            raise NotImplementedError
        return get_indice_pairs_func(
            indices,
            grid,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )


def indice_conv(
    features, filters, indice_pairs, indice_pair_num, num_activate_out, inverse=False, subm=False
):
    if filters.dtype == torch.float32:
        return sparse_conv_ext.indice_conv_fp32(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            int(inverse),
            int(subm),
        )
    elif filters.dtype == torch.half:
        return sparse_conv_ext.indice_conv_half(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            int(inverse),
            int(subm),
        )
    else:
        raise NotImplementedError


def fused_indice_conv(
    features, filters, bias, indice_pairs, indice_pair_num, num_activate_out, inverse, subm
):
    if features.dtype == torch.half:
        func = sparse_conv_ext.fused_indice_conv_half
    elif filters.dtype == torch.float32:
        func = sparse_conv_ext.fused_indice_conv_fp32
    else:
        raise NotImplementedError

    return func(
        features,
        filters,
        bias,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        int(inverse),
        int(subm),
    )


def indice_conv_backward(
    features, filters, out_bp, indice_pairs, indice_pair_num, inverse=False, subm=False
):
    if filters.dtype == torch.float32:
        return sparse_conv_ext.indice_conv_backward_fp32(
            features, filters, out_bp, indice_pairs, indice_pair_num, int(inverse), int(subm)
        )
    elif filters.dtype == torch.half:
        return sparse_conv_ext.indice_conv_backward_half(
            features, filters, out_bp, indice_pairs, indice_pair_num, int(inverse), int(subm)
        )
    else:
        raise NotImplementedError


def indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out):
    if features.dtype == torch.float32:
        return sparse_conv_ext.indice_maxpool_fp32(
            features, indice_pairs, indice_pair_num, num_activate_out
        )
    elif features.dtype == torch.half:
        return sparse_conv_ext.indice_maxpool_half(
            features, indice_pairs, indice_pair_num, num_activate_out
        )
    else:
        raise NotImplementedError


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs, indice_pair_num):
    if features.dtype == torch.float32:
        return sparse_conv_ext.indice_maxpool_backward_fp32(
            features, out_features, out_bp, indice_pairs, indice_pair_num
        )
    elif features.dtype == torch.half:
        return sparse_conv_ext.indice_maxpool_backward_half(
            features, out_features, out_bp, indice_pairs, indice_pair_num
        )
    else:
        raise NotImplementedError
class SparseConvFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx, features, filters, indice_pairs, indice_pair_num, num_activate_out
    ):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return indice_conv(
            features, filters, indice_pairs, indice_pair_num, num_activate_out, False
        )

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num, False
        )

        return input_bp, filters_bp, None, None, None


class SparseInverseConvFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx, features, filters, indice_pairs, indice_pair_num, num_activate_out
    ):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return indice_conv(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            True,
            False,
        )

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num, True, False
        )

        return input_bp, filters_bp, None, None, None


class SubMConvFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx, features, filters, indice_pairs, indice_pair_num, num_activate_out
    ):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return indice_conv(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            False,
            True,
        )

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num, False, True
        )

        return input_bp, filters_bp, None, None, None


class SparseMaxPoolFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, features, indice_pairs, indice_pair_num, num_activate_out):
        out = indice_maxpool(
            features, indice_pairs, indice_pair_num, num_activate_out
        )
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, out)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = indice_maxpool_backward(
            features, out, grad_output, indice_pairs, indice_pair_num
        )
        return input_bp, None, None, None


indice_conv = SparseConvFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_maxpool = SparseMaxPoolFunction.apply

def fused_indice_conv(
    features, filters, bias, indice_pairs, indice_pair_num, num_activate_out, inverse, subm
):
    if features.dtype == torch.half:
        func = sparse_conv_ext.fused_indice_conv_half
    elif filters.dtype == torch.float32:
        func = sparse_conv_ext.fused_indice_conv_fp32
    else:
        raise NotImplementedError

    return func(
        features,
        filters,
        bias,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        int(inverse),
        int(subm),
    )


def get_indice_pairs(
    indices,
    batch_size,
    spatial_shape,
    ksize=3,
    stride=1,
    padding=0,
    dilation=1,
    out_padding=0,
    subm=False,
    transpose=False,
    grid=None,
):
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(
                spatial_shape, ksize, stride, padding, dilation, out_padding
            )
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride, padding, dilation)

    else:
        out_shape = spatial_shape
    if grid is None:
        if ndim == 2:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_2d
        elif ndim == 3:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_3d
        elif ndim == 4:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_4d
        else:
            raise NotImplementedError
        return get_indice_pairs_func(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )
    else:
        if ndim == 2:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_grid_2d
        elif ndim == 3:
            get_indice_pairs_func = sparse_conv_ext.get_indice_pairs_grid_3d
        else:
            raise NotImplementedError
        return get_indice_pairs_func(
            indices,
            grid,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )

def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[
            i
        ] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size

def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
        output_size.append(size)
    return output_size


def _calculate_fan_in_and_fan_out_hwio(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError(
            "fan in and fan out can not be computed for tensor" "with fewer than 2 dimensions"
        )

    if dimensions == 2:  # Linear
        fan_in = tensor.size(-2)
        fan_out = tensor.size(-1)
    else:
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[..., 0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1] :]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

# class SparseModule(nn.Module):
#     """place holder, All module subclass from this will take sptensor in
#     SparseSequential."""
#
#     pass


# class SparseConvTensor:
#     def __init__(self, features, indices, spatial_shape, batch_size, grid=None):
#         """
#         Args:
#             grid: pre-allocated grid tensor.
#                   should be used when the volume of spatial shape
#                   is very large.
#         """
#         self.features = features
#         self.indices = indices
#         if self.indices.dtype != torch.int32:
#             self.indices.int()
#         self.spatial_shape = spatial_shape
#         self.batch_size = batch_size
#         self.indice_dict = {}
#         self.grid = grid
#
#     @property
#     def spatial_size(self):
#         return np.prod(self.spatial_shape)
#
#     def find_indice_pair(self, key):
#         if key is None:
#             return None
#         if key in self.indice_dict:
#             return self.indice_dict[key]
#         return None
#
#     def dense(self, channels_first=True):
#         output_shape = (
#             [self.batch_size] + list(self.spatial_shape) + [self.features.shape[1]]
#         )
#         res = scatter_nd(self.indices.long(), self.features, output_shape)
#         if not channels_first:
#             return res
#         ndim = len(self.spatial_shape)
#         trans_params = list(range(0, ndim + 1))
#         trans_params.insert(1, ndim + 1)
#         return res.permute(*trans_params).contiguous()
#
#     @property
#     def sparity(self):
#         return self.indices.shape[0] / np.prod(self.spatial_shape) / self.batch_size


class SparseConvolution(SparseModule):
    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        subm=False,
        output_padding=0,
        transposed=False,
        inverse=False,
        indice_key=None,
        fused_bn=False,
    ):
        super(SparseConvolution, self).__init__()
        assert groups == 1
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim
        if not isinstance(output_padding, (list, tuple)):
            output_padding = [output_padding] * ndim

        for d, s in zip(dilation, stride):
            assert any([s == 1, d == 1]), "don't support this."

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1x1 = np.prod(kernel_size) == 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.inverse = inverse
        self.output_padding = output_padding
        self.groups = groups
        self.subm = subm
        self.indice_key = indice_key
        self.fused_bn = fused_bn

        self.weight = Parameter(torch.Tensor(*kernel_size, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out_hwio(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        assert isinstance(input, SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            if self.transposed:
                out_spatial_shape = get_deconv_output_size(
                    spatial_shape,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.output_padding,
                )
            else:
                out_spatial_shape = get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation
                )

        else:
            out_spatial_shape = spatial_shape
        # input.update_grid(out_spatial_shape)
        # t = time.time()
        if self.conv1x1:
            features = torch.mm(
                input.features, self.weight.view(self.in_channels, self.out_channels)
            )
            if self.bias is not None:
                features += self.bias
            out_tensor = SparseConvTensor(
                features, input.indices, input.spatial_shape, input.batch_size
            )
            out_tensor.indice_dict = input.indice_dict
            out_tensor.grid = input.grid
            return out_tensor
        datas = input.find_indice_pair(self.indice_key)
        if self.inverse:
            assert datas is not None and self.indice_key is not None
            _, outids, indice_pairs, indice_pair_num, out_spatial_shape = datas
            assert indice_pairs.shape[0] == np.prod(
                self.kernel_size
            ), "inverse conv must have same kernel size as its couple conv"
        else:
            if self.indice_key is not None and datas is not None:
                outids, _, indice_pairs, indice_pair_num, _ = datas
            else:
                outids, indice_pairs, indice_pair_num = get_indice_pairs(
                    indices,
                    batch_size,
                    spatial_shape,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.output_padding,
                    self.subm,
                    self.transposed,
                    grid=input.grid,
                )
                input.indice_dict[self.indice_key] = (
                    outids,
                    indices,
                    indice_pairs,
                    indice_pair_num,
                    spatial_shape,
                )
        if self.fused_bn:
            assert self.bias is not None
            out_features = fused_indice_conv(
                features,
                self.weight,
                self.bias,
                indice_pairs.to(device),
                indice_pair_num,
                outids.shape[0],
                self.inverse,
                self.subm,
            )
        else:
            if self.subm:
                out_features = indice_subm_conv(
                    features, self.weight, indice_pairs.to(device), indice_pair_num, outids.shape[0]
                )
            else:
                if self.inverse:
                    out_features = indice_inverse_conv(
                        features,
                        self.weight,
                        indice_pairs.to(device),
                        indice_pair_num,
                        outids.shape[0],
                    )
                else:
                    out_features = indice_conv(
                        features,
                        self.weight,
                        indice_pairs.to(device),
                        indice_pair_num,
                        outids.shape[0],
                    )

            if self.bias is not None:
                out_features += self.bias
        out_tensor = SparseConvTensor(out_features, outids, out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor

def is_spconv_module(module):
    spconv_modules = (SparseModule,)
    return isinstance(module, spconv_modules)


def is_sparse_conv(module):

    return isinstance(module, SparseConvolution)



class SparseSequential(SparseModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the
    constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = SparseSequential(
                  SparseConv2d(1,20,5),
                  nn.ReLU(),
                  SparseConv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = SparseSequential(OrderedDict([
                  ('conv1', SparseConv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', SparseConv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = SparseSequential(
                  conv1=SparseConv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=SparseConv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(SparseSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)
        self._sparity_dict = {}

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    @property
    def sparity_dict(self):
        return self._sparity_dict

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            if is_spconv_module(module):  # use SpConvTensor as input
                assert isinstance(input, SparseConvTensor)
                self._sparity_dict[k] = input.sparity
                input = module(input)
            else:
                if isinstance(input, SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input.features = module(input.features)

                else:
                    input = module(input)
        return input

    def fused(self):
        """don't use this.

        no effect.
        """

        mods = [v for k, v in self._modules.items()]
        fused_mods = []
        idx = 0
        while idx < len(mods):
            if is_sparse_conv(mods[idx]):
                if idx < len(mods) - 1 and isinstance(mods[idx + 1], nn.BatchNorm1d):
                    new_module = SparseConvolution(
                        ndim=mods[idx].ndim,
                        in_channels=mods[idx].in_channels,
                        out_channels=mods[idx].out_channels,
                        kernel_size=mods[idx].kernel_size,
                        stride=mods[idx].stride,
                        padding=mods[idx].padding,
                        dilation=mods[idx].dilation,
                        groups=mods[idx].groups,
                        bias=True,
                        subm=mods[idx].subm,
                        output_padding=mods[idx].output_padding,
                        transposed=mods[idx].transposed,
                        inverse=mods[idx].inverse,
                        indice_key=mods[idx].indice_key,
                        fused_bn=True,
                    )
                    new_module.load_state_dict(mods[idx].state_dict(), False)
                    new_module.to(mods[idx].weight.device)
                    conv = new_module
                    bn = mods[idx + 1]
                    conv.bias.data.zero_()
                    conv.weight.data[:] = (
                        conv.weight.data * bn.weight.data / (torch.sqrt(bn.running_var) + bn.eps)
                    )
                    conv.bias.data[:] = (conv.bias.data - bn.running_mean) * bn.weight.data / (
                        torch.sqrt(bn.running_var) + bn.eps
                    ) + bn.bias.data
                    fused_mods.append(conv)
                    idx += 2
                else:
                    fused_mods.append(mods[idx])
                    idx += 1
            else:
                fused_mods.append(mods[idx])
                idx += 1
        return SparseSequential(*fused_mods)



@MIDDLE_ENCODERS.register_module()
class CustomSparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
    """

    def __init__(
        self,
        in_channels,
        sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
    ):
        super().__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, (list, tuple)) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        if self.order[0] != "conv":  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
                order=("conv",),
            )
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
            )

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, norm_cfg, self.base_channels, block_type=block_type
        )

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(2, 1, 1),
            stride=(1, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="spconv_down2",
            conv_type="SparseConv3d",
        )

    @auto_fp16(apply_to=("voxel_features",))
    def forward(self, voxel_features, coors, batch_size, **kwargs):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features

    def make_encoder_layers(
        self,
        make_block,
        norm_cfg,
        in_channels,
        block_type="conv_module",
        conv_cfg=dict(type="SubMConv3d"),
    ):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ["conv_module", "basicblock"]
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="SparseConv3d",
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f"spconv{i + 1}",
                                conv_type="SparseConv3d",
                            )
                        )
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_type="SubMConv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels



@DETECTORS.register_module()
class MergedTopoNetMapGraphLidar(MVXTwoStageDetector):

    def __init__(self,
                 bev_constructor=None,
                 bbox_head=None,
                 lane_head=None,
                 map_encoder=None,
                 video_test_mode=False,
                 use_grid_mask=False,
                 modality='vision',
                 lidar_encoder=None,
                 **kwargs):

        super(MergedTopoNetMapGraphLidar, self).__init__(**kwargs)

        if map_encoder is not None:
            self.map_encoder_type = map_encoder['type']
            self.map_encoder = build_neck(map_encoder)

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

        # Add lidar
        self.modality = 'fusion'
        if self.modality == 'fusion' and lidar_encoder is not None:
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": builder.build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)

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

        # bias initialization: initialize to ones
        # nn.init.constant_(self.rel_predictor_gate_tecl.bias, 1.0)
        # nn.init.constant_(self.rel_predictor_gate_clcl.bias, 1.0)

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

    def extract_map_feat(self, map_graph, **kwargs):
        """Extract features from images and points."""
        sd_map_feats = self.map_encoder(map_graph, **kwargs)
        # list batch, num_polylines * feat_dim
        return sd_map_feats

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

    def predict(self, img_feats, img_metas, prev_bev, front_view_img_feats, bbox_img_metas, map_graph=None, map_x=None, lidar_feat=None, **kwargs):

        # 1. prepare inputs
        te_feats = None
        te_cls_scores = None

        # bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)
        # map_graph_feats is already provided, get bev_feats directly
        if map_x is not None:
            bev_feats = self.bev_constructor(img_feats, img_metas, map_graph_feats=map_x, prev_bev=prev_bev, lidar_feat=lidar_feat)
        # get bev features of SD map raster
        elif map_graph is not None:
            map_graph_feats = self.extract_map_feat(map_graph=map_graph, **kwargs)
            bev_feats = self.bev_constructor(img_feats, img_metas, map_graph_feats=map_graph_feats, prev_bev=prev_bev, lidar_feat=lidar_feat)

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
        device = decoder_attention_queries_te[0].device
        # For TE
        QK_te_shape = [6, 100, 1, 256]
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
        QK_cl_shape = [6, 200, 1, 256]
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
        sequence_output_clcl = query_cl.permute(1, 0,2)
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
        subject_output_tecl = self.final_sub_proj_tecl(sequence_output_tecl).unsqueeze(2).expand(-1, -1, num_object_queries, -1)
        object_output_tecl = self.final_obj_proj_tecl(sequence_output_tecl).unsqueeze(1).expand(-1, num_object_queries, -1, -1)
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
        outputs_te_transformer_second_half = self.bbox_head.transformer.forward_second_half(*outputs_te_decoder, outputs_te_transformer_first_half['init_reference_out'])
        bbox_outs = self.bbox_head.forward_second_half(*outputs_te_transformer_second_half)
        outputs_cl_transformer_last_half = self.pts_bbox_head.transformer.forward_second_half(*outputs_cl_decoder, outputs_cl_transformer_first_half['init_reference_out'])
        outs = self.pts_bbox_head.forward_second_half(outputs_cl_transformer_last_half)

        outs['all_lclc_preds'] = [pred_connectivity_clcl]
        outs['all_lcte_preds'] = [pred_connectivity_tecl]

        return bbox_outs, outs, bev_feats

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=('points'), out_fp32=True)
    def extract_lidar_feat(self, points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)

        return lidar_feat

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
                      points=None,
                      **kwargs
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

        # Extract lidar features
        lidar_feat = None
        if self.modality == 'fusion':
            lidar_feat = self.extract_lidar_feat(points)

        # 2. Generate predictions
        bbox_outs, outs, _ = self.predict(img_feats, img_metas, prev_bev, front_view_img_feats, bbox_img_metas, map_graph=map_graph, lidar_feat=lidar_feat, **kwargs)

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

    def forward_test(self, img_metas, img=None, map_graph=None, points=None, **kwargs):
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
            img_metas, img, map_graph, prev_bev=self.prev_frame_info['prev_bev'], points=points, **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return results_list

    def simple_test_pts(self, x, map_x, img_metas, img=None, prev_bev=None, points=None,
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

        lidar_feat = None
        if self.modality == 'fusion':
            lidar_feat = self.extract_lidar_feat(points)

        # 2. Generate predictions
        bbox_outs, outs, bev_feats = self.predict(x, img_metas, prev_bev, front_view_img_feats, bbox_img_metas, map_x=map_x, lidar_feat=lidar_feat)

        # 3. Get boxes, lanes, and relations
        bbox_results = self.bbox_head.get_bboxes(bbox_outs, bbox_img_metas, rescale=rescale)
        lane_results, lclc_results, lcte_results = self.pts_bbox_head.get_lanes(outs, img_metas, rescale=rescale)

        return bev_feats, bbox_results, lane_results, lclc_results, lcte_results

    def simple_test(self, img_metas, img=None, map_graph=None, prev_bev=None, rescale=False, points=None, **kwargs):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # get bev features of SD map raster
        map_graph_feats = self.extract_map_feat(map_graph=map_graph, **kwargs)

        results_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_results, lane_results, lclc_results, lcte_results = self.simple_test_pts(
            img_feats, map_graph_feats, img_metas, img, prev_bev, points=points, rescale=rescale)

        ########################################################################
        if False:
            import torch
            import numpy as np
            from plyfile import PlyData, PlyElement
            # Assuming points[0] is your tensor
            points = points[0].cpu().numpy()  # Move to CPU and convert to NumPy array
            # Create vertices for PLY format
            vertices = np.array([tuple(point) for point in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            # Create PLY element
            ply_element = PlyElement.describe(vertices, 'vertex')
            # Write to PLY file
            ply_file_path = 'debug_lidar/current_points.ply'
            PlyData([ply_element]).write(ply_file_path)
        ########################################################################

        for result_dict, bbox, lane, lclc, lcte in zip(results_list, bbox_results, lane_results, lclc_results,
                                                       lcte_results):
            result_dict['bbox_results'] = bbox
            result_dict['lane_results'] = lane
            result_dict['lclc_results'] = lclc
            result_dict['lcte_results'] = lcte
        return new_prev_bev, results_list
