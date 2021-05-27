_base_ = [
    '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
albu_train_transforms = [
    dict(
        type="Cutout",
        num_holes=30,
        max_h_size=30,
        max_w_size=30,
        fill_value=img_norm_cfg["mean"][::-1],
        p=0.1,
    ),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0),
            dict(type="RandomGamma"),
            dict(type="CLAHE"),
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type="MotionBlur"),
            dict(type="GaussNoise"),
            dict(type="ImageCompression", quality_lower=75),
        ],
        p=0.1),
    dict(type='RandomRotate90', p=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

# _base_ = [
#     '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
#     '../_base_/datasets/coco_instance.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]

# model = dict(
#     backbone=dict(
#         embed_dim=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         window_size=7,
#         ape=False,
#         drop_path_rate=0.3,
#         patch_norm=True,
#         use_checkpoint=False
#     ),
#     neck=dict(in_channels=[128, 256, 512, 1024]),
#     roi_head=dict(
#         bbox_head=[
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=80,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=80,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=80,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
#         ]))

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



# # augmentation strategy originates from DETR / Sparse RCNN
# # train_pipeline = [
# #     dict(type='LoadImageFromFile'),
# #     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
# #     dict(type='RandomFlip', flip_ratio=0.5),
# #     dict(type='AutoAugment',
# #          policies=[
# #              [
# #                  dict(type='Resize',
# #                       img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
# #                                  (608, 1333), (640, 1333), (672, 1333), (704, 1333),
# #                                  (736, 1333), (768, 1333), (800, 1333)],
# #                       multiscale_mode='value',
# #                       keep_ratio=True)
# #              ],
# #              [
# #                  dict(type='Resize',
# #                       img_scale=[(400, 1333), (500, 1333), (600, 1333)],
# #                       multiscale_mode='value',
# #                       keep_ratio=True),
# #                  dict(type='RandomCrop',
# #                       crop_type='absolute_range',
# #                       crop_size=(384, 600),
# #                       allow_negative_crop=True),
# #                  dict(type='Resize',
# #                       img_scale=[(480, 1333), (512, 1333), (544, 1333),
# #                                  (576, 1333), (608, 1333), (640, 1333),
# #                                  (672, 1333), (704, 1333), (736, 1333),
# #                                  (768, 1333), (800, 1333)],
# #                       multiscale_mode='value',
# #                       override=True,
# #                       keep_ratio=True)
# #              ]
# #          ]),
# #     dict(type='Normalize', **img_norm_cfg),
# #     dict(type='Pad', size_divisor=32),
# #     dict(type='DefaultFormatBundle'),
# #     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# # ]

# # our training augmentation
# albu_train_transforms = [
#     dict(
#         type="Cutout",
#         num_holes=30,
#         max_h_size=30,
#         max_w_size=30,
#         fill_value=img_norm_cfg["mean"][::-1],
#         p=0.1,
#     ),
#     dict(
#         type='RandomBrightnessContrast',
#         brightness_limit=[0.1, 0.3],
#         contrast_limit=[0.1, 0.3],
#         p=0.2),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(
#                 type='RGBShift',
#                 r_shift_limit=10,
#                 g_shift_limit=10,
#                 b_shift_limit=10,
#                 p=1.0),
#             dict(
#                 type='HueSaturationValue',
#                 hue_shift_limit=20,
#                 sat_shift_limit=30,
#                 val_shift_limit=20,
#                 p=1.0),
#             dict(type="RandomGamma"),
#             dict(type="CLAHE"),
#         ],
#         p=0.1),
#     dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
#     dict(type='ChannelShuffle', p=0.1),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='Blur', blur_limit=3, p=1.0),
#             dict(type='MedianBlur', blur_limit=3, p=1.0),
#             dict(type="MotionBlur"),
#             dict(type="GaussNoise"),
#             dict(type="ImageCompression", quality_lower=75),
#         ],
#         p=0.1),
#     dict(type='RandomRotate90', p=0.2),
# ]

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(
#         type='Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_labels'],
#             min_visibility=0.0,
#             filter_lost_elements=True),
#         keymap={
#             'img': 'image',
#             'gt_masks': 'masks',
#             'gt_bboxes': 'bboxes'
#         },
#         update_pad_shape=False,
#         skip_img_without_anno=True
#     )
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]


# data = dict(train=dict(pipeline=train_pipeline))

# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))
# lr_config = dict(step=[27, 33])

# # apex 사용 시
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# # apex 사용 안할 때
# # runner = dict(type='EpochBasedRunner', max_epochs=36)

# # do not use mmdet version fp16
# # apex 사용하면 사용
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
# load_from = "/opt/ml/code/SwinModel/cascade_mask_rcnn_swin_base_patch4_window7.pth"
