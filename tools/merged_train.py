# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Tianyu Li
# ---------------------------------------------
from __future__ import division

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

import functools
import inspect
import typing as t

from torch import nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

class SpectralNormedWeight(nn.Module):
    """SpectralNorm Layer. First sigma uses SVD, then power iteration."""

    def __init__(
        self,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.weight = weight
        with torch.no_grad():
            _, s, vh = torch.linalg.svd(self.weight, full_matrices=False)

        self.register_buffer("u", vh[0])
        self.register_buffer("spectral_norm", s[0] * torch.ones(1))

    def get_sigma(self, u: torch.Tensor, weight: torch.Tensor):
        with torch.no_grad():
            v = weight.mv(u)
            v = nn.functional.normalize(v, dim=0)
            u = weight.T.mv(v)
            u = nn.functional.normalize(u, dim=0)
            if self.training:
                self.u.data.copy_(u)

        return torch.einsum("c,cd,d->", v, weight, u)

    def forward(self):
        """Normalize by largest singular value and rescale by learnable."""
        sigma = self.get_sigma(u=self.u, weight=self.weight)
        if self.training:
            self.spectral_norm.data.copy_(sigma)

        return self.weight / sigma


class FP32SpectralNormedWeight(nn.Module):
    """SpectralNorm FP32 wrapper."""

    __constants__ = ["enabled"]  # for jit-scripting

    def __init__(self, module: nn.Module, enabled: bool = True):
        super().__init__()
        self.net = module
        self.enabled = enabled

    def __repr__(self):
        """Extra str info."""
        return (
            f"FP32SpectralNormedWeight({self.net.__repr__()}, enabled={self.enabled})"
        )

    def forward(self):
        with torch.cuda.amp.autocast(enabled=self.enabled):
            u = self.net.u
            weight = self.net.weight

            if not self.enabled:
                u = u.float()
                weight = weight.float()

            sigma = self.net.get_sigma(u=u, weight=weight)
            if self.training:
                self.net.spectral_norm.data.copy_(sigma)

            return weight / sigma

    @property
    def spectral_norm(self) -> torch.Tensor:
        return self.net.spectral_norm


class SNLinear(nn.Linear):
    """Spectral Norm linear from sigmaReparam.

    Optionally, if 'stats_only' is `True`,then we
    only compute the spectral norm for tracking
    purposes, but do not use it in the forward pass.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_multiplier: float = 1.0,
        stats_only: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.stats_only = stats_only
        self.init_multiplier = init_multiplier

        self.init_std = 0.02 * init_multiplier
        nn.init.trunc_normal_(self.weight, std=self.init_std)

        # Handle normalization and add a learnable scalar.
        self.spectral_normed_weight = SpectralNormedWeight(self.weight)
        sn_init = self.spectral_normed_weight.spectral_norm

        # Would have set sigma to None if `stats_only` but jit really disliked this
        self.sigma = (
            torch.ones_like(sn_init)
            if self.stats_only
            else nn.Parameter(
                torch.zeros_like(sn_init).copy_(sn_init), requires_grad=True
            )
        )

        self.register_buffer("effective_spectral_norm", sn_init)
        self.update_effective_spec_norm()

        #: TODO: make sure this is correct
        del self.weight


    def update_effective_spec_norm(self):
        """Update the buffer corresponding to the spectral norm for tracking."""
        with torch.no_grad():
            s_0 = (
                self.spectral_normed_weight.spectral_norm
                if self.stats_only
                else self.sigma
            )
            self.effective_spectral_norm.data.copy_(s_0)

    def get_weight(self):
        """Get the reparameterized or reparameterized weight matrix depending on mode
        and update the external spectral norm tracker."""
        normed_weight = self.spectral_normed_weight()
        self.update_effective_spec_norm()
        return self.weight if self.stats_only else normed_weight * self.sigma

    def forward(self, inputs: torch.Tensor):
        weight = self.get_weight()
        return F.linear(inputs, weight, self.bias)


class SNConv2d(SNLinear):
    """Spectral norm based 2d conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: t.Union[int, t.Iterable[int]],
        stride: t.Union[int, t.Iterable[int]] = 1,
        padding: t.Union[int, t.Iterable[int]] = 0,
        dilation: t.Union[int, t.Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # NB(jramapuram): not used
        init_multiplier: float = 1.0,
        stats_only: bool = False,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        in_features = in_channels * kernel_size[0] * kernel_size[1]
        super().__init__(
            in_features,
            out_channels,
            bias=bias,
            init_multiplier=init_multiplier,
            stats_only=stats_only,
        )

        assert padding_mode == "zeros"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.stats_only = stats_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.get_weight()
        weight = weight.view(
            self.out_features, -1, self.kernel_size[0], self.kernel_size[1]
        )
        return F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def convert_layer(
    container: nn.Module,
    from_layer: t.Callable,
    to_layer: t.Callable,
    set_from_layer_kwargs: bool = True,
    ignore_from_signature: t.Optional[t.Iterable] = None,
) -> nn.Module:
    """Convert from_layer to to_layer for all layers in container.

    :param container: torch container, nn.Sequential, etc.
    :param from_layer: a class definition (eg: nn.Conv2d)
    :param to_layer: a class defition (eg: GatedConv2d)
    :param set_from_layer_kwargs: uses the kwargs from from_layer and set to_layer values
    :param ignore_from_signature: ignore these fields from signature matching
    :returns: nn.Module

    """
    for child_name, child in container.named_children():
        if isinstance(child, from_layer):
            to_layer_i = to_layer
            if set_from_layer_kwargs:
                signature_list = inspect.getfullargspec(from_layer).args[
                    1:
                ]  # 0th element is arg-list, 0th of that is 'self'
                if ignore_from_signature is not None:
                    signature_list = [
                        k for k in signature_list if k not in ignore_from_signature
                    ]

                kwargs = {
                    sig: getattr(child, sig)
                    if sig != "bias"
                    else bool(child.bias is not None)
                    for sig in signature_list
                }
                to_layer_i = functools.partial(to_layer, **kwargs)

            setattr(container, child_name, to_layer_i())
        else:
            convert_layer(
                child,
                from_layer,
                to_layer,
                set_from_layer_kwargs=set_from_layer_kwargs,
                ignore_from_signature=ignore_from_signature,
            )

    return container


def convert_to_sn(
    network: nn.Module, linear_init_gain: float = 1.0, conv_init_gain: float = 1.0
) -> nn.Module:
    """Convert Linear and Conv2d layers to their SigmaReparam equivalents.

    :param network: The container to convert on.
    :param linear_init_gain: trunc_norm(0, 0.02 * linear_init_gain) for Linear
    :param conv_init_gain: trunc_norm(0, 0.02 * conv_init_gain) for Conv2d

    """
    layers_for_conversion = [
        {
            "name": "Linear",
            "from": nn.Linear,
            "to": functools.partial(SNLinear, init_multiplier=linear_init_gain),
        },
        {
            "name": "Conv2d",
            "from": nn.Conv2d,
            "to": functools.partial(SNConv2d, init_multiplier=conv_init_gain),
        },
    ]  # Layers need to be in this order so that Linear is converted before Conv2d.

    for layer in layers_for_conversion:
        convert_layer(
            container=network,
            from_layer=layer["from"],
            to_layer=layer["to"],
            set_from_layer_kwargs=True,
            ignore_from_signature=("device", "dtype"),
        )

    return network


NORMALIZATION_LAYER_TYPE_MAP = {
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "GroupNorm": nn.GroupNorm,
    "InstanceNorm1d": nn.InstanceNorm1d,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "InstanceNorm3d": nn.InstanceNorm3d,
    "LayerNorm": nn.LayerNorm,
}


def remove_all_normalization_layers(network: nn.Module) -> nn.Module:
    """Replaces normalization layers with Identity."""
    for layer_name, layer_type in NORMALIZATION_LAYER_TYPE_MAP.items():
        print(f"Removing Normalization Layer '{layer_name}' with type {layer_type}")
        convert_layer(
            container=network,
            from_layer=layer_type,
            to_layer=nn.Identity,
            set_from_layer_kwargs=False,
        )

    return network


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation version >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        if 'auto_scale_lr' in cfg and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file.')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name='mmdet')

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # Remove unused layers
    for i in range(6):
        model.pts_bbox_head.transformer.decoder.layers[i].attentions[0] = None

    model.pts_bbox_head.te_embed_branches = None

    for i in range(len(model.pts_bbox_head.lclc_branches) - 1):
        model.pts_bbox_head.lclc_branches[i] = None
        model.pts_bbox_head.lcte_branches[i] = None
    model.pts_bbox_head.lclc_branches[-1].MLP_o1 = None
    model.pts_bbox_head.lclc_branches[-1].MLP_o2 = None
    model.pts_bbox_head.lclc_branches[-1].classifier = None
    model.pts_bbox_head.lcte_branches[-1].MLP_o1 = None
    model.pts_bbox_head.lcte_branches[-1].MLP_o2 = None
    model.pts_bbox_head.lcte_branches[-1].classifier = None
    # cfg.find_unused_parameters = True

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    auto_scale_lr(cfg, distributed=distributed, logger=logger)


    # TODO: sigma-reparam
    # TODO: make sure to address: "unexpected key in source state_dict: fc.weight, fc.bias"
    original_img_backbone = copy.deepcopy(model.img_backbone)
    model = remove_all_normalization_layers(convert_to_sn(model, conv_init_gain=0.125))
    model.img_backbone = original_img_backbone

    # TODO: deactivate warmup and some other things

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()