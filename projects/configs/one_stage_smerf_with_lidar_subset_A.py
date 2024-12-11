_base_ = []
custom_imports = dict(imports=['projects.bevformer', 'projects.toponet'])

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]
lidar_point_cloud_range = point_cloud_range # 1024, 512, 20
# lidar_point_cloud_range = [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0]

voxel_size = [0.1, 0.1, 0.2] # TODO: check if need to modify

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ['centerline']

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
num_cams = 7
pts_dim = 3

dataset_type = 'OpenLaneV2_subset_A_GraphicalSDMapDataset'
data_root = 'data/OpenLane-V2/'

para_method = 'fix_pts_interp'
method_para = dict(n_points=11)
code_size = pts_dim * method_para['n_points']
sd_method_para = dict(n_points=11)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_ffn_cfg_ = dict(
    type='FFN',
    embed_dims=_dim_,
    feedforward_channels=_ffn_dim_,
    num_fcs=2,
    ffn_drop=0.1,
    act_cfg=dict(type='ReLU', inplace=True),
),

_num_levels_ = 4
_num_heads_ = 4
bev_h_ = 100
bev_w_ = 200

model = dict(
    type='MergedTopoNetMapGraphLidar',
    use_grid_mask=True,
    video_test_mode=False,
    modality='fusion',
    lidar_encoder=dict(
        voxelize=dict(max_num_points=10,point_cloud_range=lidar_point_cloud_range,voxel_size=voxel_size,max_voxels=[90000, 120000]),
        backbone=dict(
            type='CustomSparseEncoder',
            in_channels=3,
            sparse_shape=[20, 512, 1024],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=([1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1],
                              [1, 1]),
            block_type='basicblock'
        ),
    ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    map_encoder=dict(
        type='MapGraphTransformer',
        input_dim=360,  # 32 * 11 + 8
        dmodel=_dim_,      # TODO: maybe change?
        hidden_dim=_dim_,  # TODO: maybe change?
        nheads=_num_heads_,
        nlayers=6,
        batch_first=True,
        pos_encoder=dict(
            type='SineContinuousPositionalEncoding',
            num_feats=16,  # 2 * 16 = 32 final dim
            temperature=1000,
            normalize=True,
            range=[point_cloud_range[3] - point_cloud_range[0], point_cloud_range[4] - point_cloud_range[1]],
            offset=[point_cloud_range[0], point_cloud_range[1]],
        ),
    ),
    bev_constructor=dict(
        type='BEVFormerConstructer',
        num_feature_levels=_num_levels_,
        num_cams=num_cams,
        embed_dims=_dim_,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        rotate_center=[bev_h_//2, bev_w_//2],
        fuser=dict(
            type='ConvFuser',
            in_channels=[_dim_, 256],
            out_channels=_dim_,
        ),
        encoder=dict(
            type='BEVFormerEncoder',
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalSelfAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='SpatialCrossAttention',
                        embed_dims=_dim_,
                        num_cams=num_cams,
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=8,
                            num_levels=_num_levels_)
                    ),
                    dict(
                        type='MaskedCrossAttention',
                        embed_dims=_dim_,
                        num_heads=_num_heads_,),

                ],
                ffn_cfgs=_ffn_cfg_,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'cross_attn_graph', 'norm',
                                 'ffn', 'norm'))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
    ),
    bbox_head=dict(
        type='CustomDeformableDETRHead',
        num_query=100,
        num_classes=13,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='MyDeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=_dim_),
                    ffn_cfgs=_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='MyDeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MyDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MyMultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_pos_dim_,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.5),
        loss_iou=dict(type='GIoULoss', loss_weight=1.0),
        test_cfg=dict(max_per_img=100)),
    lane_head=dict(
        type='TopoNetHead',
        num_classes=1,
        in_channels=_dim_,
        num_query=200,
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        pts_dim=pts_dim,
        sync_cls_avg_factor=False,
        code_size=code_size,
        code_weights= [1.0 for i in range(code_size)],
        transformer=dict(
            type='MyTopoNetTransformerDecoderOnly',
            embed_dims=_dim_,
            pts_dim=pts_dim,
            decoder=dict(
                type='MyTopoNetSGNNDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MySGNNDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MyMultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    ffn_cfgs=dict(
                        type='MyDummy_FFN_SGNN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_te_classes=13,
                        edge_weight=0.6),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        lclc_head=dict(
            type='SingleLayerRelationshipHead',
            in_channels_o1=_dim_,
            in_channels_o2=_dim_,
            shared_param=False,
            loss_rel=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=5)),
        lcte_head=dict(
            type='SingleLayerRelationshipHead',
            in_channels_o1=_dim_,
            in_channels_o2=_dim_,
            shared_param=False,
            loss_rel=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=5)),
        bbox_coder=dict(type='LanePseudoCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.025)),
    # model training and testing settings
    train_cfg=dict(
        bbox=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='BBoxL1Cost', weight=2.5, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0))),
        lane=dict(
            assigner=dict(
                type='LaneHungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=1.5),
                reg_cost=dict(type='LaneL1Cost', weight=0.025),
                pc_range=point_cloud_range))))

reduce_beams=32
# load_dim=5
# use_dim=5
load_dim=3
use_dim=3

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFilesToponet', to_float32=True),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=1, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='CustomPointsRangeFilter', point_cloud_range=lidar_point_cloud_range),
    dict(type='LoadAnnotations3DLane',
         with_lane_3d=True, with_lane_label_3d=True, with_lane_adj=True,
         with_bbox=True, with_label=True, with_lane_lcte_adj=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='CropFrontViewImageForAv2'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImageSame2Max', size_divisor=32),
    dict(type='GridMaskMultiViewImage'),
    dict(type='LaneParameterize3D', method=para_method, method_para=method_para),
    dict(type='CustomParametrizeSDMapGraph', method='even_points_onehot_type', method_para=sd_method_para),
    dict(type='CustomFormatBundle3DLane', class_names=class_names),
    dict(type='CustomCollect3D', keys=[
        'img',  'map_graph', 'onehot_category',
        'gt_lanes_3d', 'gt_lane_labels_3d', 'gt_lane_adj',
        'gt_bboxes', 'gt_labels', 'gt_lane_lcte_adj', 'points'])
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFilesToponet', to_float32=True),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=1, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='CustomPointsRangeFilter', point_cloud_range=lidar_point_cloud_range),
    dict(type='CropFrontViewImageForAv2'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImageSame2Max', size_divisor=32),
    dict(type='CustomParametrizeSDMapGraph', method='even_points_onehot_type', method_para=sd_method_para),
    dict(type='CustomFormatBundle3DLane', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'map_graph', 'onehot_category', 'points'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_train.pkl',
        # ann_file=data_root + 'data_dict_sample_train.pkl',
        map_dir_prefix='sd_map_graph_all',
        map_file_ext='pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        split='train',
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_val.pkl',
        # ann_file=data_root + 'data_dict_sample_train.pkl',
        map_dir_prefix='sd_map_graph_all',
        map_file_ext='pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_val.pkl',
        # ann_file=data_root + 'data_dict_sample_train.pkl',
        map_dir_prefix='sd_map_graph_all',
        map_file_ext='pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        test_mode=True)
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),

            'proj_q_te': dict(lr_mult=1.0),
            'proj_k_te': dict(lr_mult=1.0),
            'proj_q_cl': dict(lr_mult=1.0),
            'proj_k_cl': dict(lr_mult=1.0),

            'final_sub_proj_clcl': dict(lr_mult=1.0),
            'final_obj_proj_clcl': dict(lr_mult=1.0),
            'final_sub_proj_tecl': dict(lr_mult=1.0),
            'final_obj_proj_tecl': dict(lr_mult=1.0),

            'rel_predictor_gate_tecl': dict(lr_mult=1.0),
            'rel_predictor_gate_clcl': dict(lr_mult=1.0),

            'connectivity_layer_tecl': dict(lr_mult=1.0),
            'connectivity_layer_clcl': dict(lr_mult=1.0),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
