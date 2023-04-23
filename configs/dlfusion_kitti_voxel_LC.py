_base_ = [
    '_base_/datasets/kitti-3d-3class.py',
    '_base_/schedules/cyclic_40e.py', '_base_/default_runtime.py'
]

# point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]  # for nuscenes
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # for kitti
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.05, 0.05, 0.1]

# ========config for the img voxel proj==========
bev_stride = 4
cam_voxel_size = [0.15, 0.15, 8]
sample_num = 5
cam_sweep_num = 1
fp16_enabled = True
voxel_shape = [int(((point_cloud_range[3] - point_cloud_range[0]) / cam_voxel_size[0]) // bev_stride),
               int(((point_cloud_range[4] - point_cloud_range[1]) / cam_voxel_size[1]) // bev_stride),
               sample_num]
# ===============================================

out_size_factor = 8
evaluation = dict(interval=1)
dataset_type = 'KittiDataset'
# data_root = 'data/nuscenes/'
# input_modality = dict(
#     use_lidar=True,
#     use_camera=True,
#     use_radar=False,
#     use_map=False,
#     use_external=False)
img_scale = (800, 448)
num_views = 1
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


# dataset settings
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
# PointPillars adopted a different sampling strategies among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10))

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadMultiViewImageFromFiles'),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925 * 2, 0.3925 * 2],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[0.5, 0.5, 0.5]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=True,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
            dict(type='MyNormalize', **img_norm_cfg),
            dict(type='MyPad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]


model = dict(
    type='DLFusionDetector',
    freeze_img=True,
    # img_backbone=dict(
    #     type='DLASeg',
    #     num_layers=34,
    #     heads={},
    #     head_convs=-1,
    #     ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',  # mean-operation for single voxel cell
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoderHD',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=256,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
        fp16_enabled=False),  # not enable FP16 here
    pts_backbone=dict(
        type='SECOND3D',
        in_channels=[256, 256, 256],
        out_channels=[128, 256, 512],
        layer_nums=[5, 5, 5],
        layer_strides=[1, 2, 4],
        is_cascade=False,
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv3d', kernel=(1, 3, 3), bias=False)),
    pts_neck=dict(
        type='SECOND3DFPN',
        in_channels=[128, 256, 512],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv3d', bias=False),
        extra_conv=dict(type='Conv3d', num_conv=3, bias=False),
    ),
    imgvoxel_lidar_fusion=dict(
        type='VoxelFusion',
        unified_conv=dict(type='Conv3d', num_conv=1),
        view_cfg=dict(
            num_cams=6,
            num_convs=3,
            num_points=sample_num,
            num_sweeps=cam_sweep_num,
            kernel_size=(3, 3, 3),
            keep_sweep_dim=True,
            num_feature_levels=4,
            embed_dims=256,
            pc_range=point_cloud_range,
            voxel_shape=voxel_shape,
            fp16_enabled=fp16_enabled,
        ),
        depth_head=dict(
            type='SimpleDepth',
            in_channels=256,
            out_channels=256,
            model=dict(
                depth_dim=64,
            ),
        ),
        collapse_pts=True,
    ),
    _bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -1.78, 70.4, 39.68, -1.78],
            ],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))



# optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(10, 0.0001),
#     cyclic_times=1,
#     step_ratio_up=0.4)
# momentum_config = dict(
#     policy='cyclic',
#     target_ratio=(0.8947368421052632, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4)
# total_epochs = 6
# checkpoint_config = dict(interval=1)
# log_config = dict(
#     interval=50,
#     hooks=[dict(type='TextLoggerHook'),
#            dict(type='TensorboardLoggerHook')])
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# work_dir = None
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
# gpu_ids = range(0, 8)
# freeze_lidar_components = True
# find_unused_parameters = True


# optimizer
lr = 0.001
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=2)
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
total_epochs = 80

