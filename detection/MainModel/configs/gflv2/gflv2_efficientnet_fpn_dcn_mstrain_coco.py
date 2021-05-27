_base_ = [
    '../_base_/datasets/coco_detection_mstrain_480_960.py',
#     '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='GFL',
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone = dict(
        type='EfficientNet',
        model_type='efficientnet-b4',  # Possible types: ['efficientnet-b0' ... 'efficientnet-b7']
        out_indices=(0, 1, 3, 5)),  # Possible indices: [0 1 2 3 4 5 6],
    neck=dict(
        type='FPN',
        in_channels=[32,56,160,448,1792],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        use_dgqp=True,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16])
runner = dict(type='EpochBasedRunner', max_epochs=24)
