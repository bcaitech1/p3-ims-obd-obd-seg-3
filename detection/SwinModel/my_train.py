from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# config file 들고오기
# cfg = Config.fromfile('/opt/ml/SwinModel/configs/swin/siwn_one.py')
cfg = Config.fromfile('/opt/ml/SwinModel/configs/swin/new_our_swin.py')
# cfg = Config.fromfile('/opt/ml/SwinModel/configs/swin/noapex.py')

cfg.work_dir = 'work_dirs/swin2_1team'

# PREFIX = '/opt/ml/input/data/'

cfg.data.train.classes = classes
# cfg.data.train.img_prefix = PREFIX
# cfg.data.train.ann_file = PREFIX + 'train_all.json'

PREFIX = '/opt/ml/input/data/'

cfg.data.val.classes = classes
cfg.data.val.img_prefix = PREFIX
cfg.data.val.ann_file = PREFIX + 'val.json'

cfg.data.test.classes = classes
cfg.data.test.img_prefix = PREFIX
cfg.data.test.ann_file = PREFIX + 'test.json'

cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 4

cfg.seed=2020
# cfg.gpu_ids = [0]

cfg.lr_config = dict(policy='CosineAnnealing',warmup='linear',warmup_iters=3000,
                    warmup_ratio=0.0001, min_lr_ratio=1e-7)

cfg.load_from = "cascade_mask_rcnn_swin_small_patch4_window7.pth" # 얘는 pretrain 모델 가져오는 경로

cfg.gpu_ids = [0]




# dataset 바꾸기
cfg.data.train.classes = classes
cfg.data.train.img_prefix = PREFIX
cfg.data.train.ann_file = PREFIX + 'train_all.json'
cfg.data.train.seg_prefix=PREFIX                        ### 요놈 ###


cfg.data.train=dict(
        type='CocoDataset',
        ann_file='/opt/ml/input/data/train_all.json',
        img_prefix='/opt/ml/input/data/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='InstaBoost',
                action_candidate=('normal', 'horizontal', 'skip'),
                action_prob=(1, 0, 0),
                scale=(0.8, 1.2),
                dx=15,
                dy=15,
                theta=(-1, 1),
                color_prob=0.0,
                hflag=False,
                aug_ratio=0.5),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                with_seg=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type':
                    'Resize',
                    'img_scale':
                    [(512, 512), (576, 576), (640, 640), (704, 704),
                     (768, 768), (832, 832), (896, 896), (960, 960),
                     (1024, 1024)],
                    'multiscale_mode':
                    'value',
                    'keep_ratio':
                    True
                }],
                          [{
                              'type': 'Resize',
                              'img_scale': [(512, 512), (768, 768),
                                            (1024, 1024)],
                              'multiscale_mode': 'value',
                              'keep_ratio': True
                          }, {
                              'type': 'RandomCrop',
                              'crop_type': 'absolute_range',
                              'crop_size': (512, 512),
                              'allow_negative_crop': True
                          }, {
                              'type':
                              'Resize',
                              'img_scale': [(512, 512), (576, 576), (640, 640),
                                            (704, 704), (768, 768), (832, 832),
                                            (896, 896), (960, 960),
                                            (1024, 1024)],
                              'multiscale_mode':
                              'value',
                              'override':
                              True,
                              'keep_ratio':
                              True
                          }]]),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='Cutout',
                        num_holes=30,
                        max_h_size=30,
                        max_w_size=30,
                        fill_value=[103.53, 116.28, 123.675],
                        p=0.1),
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
                            dict(type='RandomGamma'),
                            dict(type='CLAHE')
                        ],
                        p=0.1),
                    dict(
                        type='JpegCompression',
                        quality_lower=85,
                        quality_upper=95,
                        p=0.2),
                    dict(type='ChannelShuffle', p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0),
                            dict(type='MotionBlur'),
                            dict(type='GaussNoise'),
                            dict(type='ImageCompression', quality_lower=75)
                        ],
                        p=0.1),
                    dict(type='RandomRotate90', p=0.5)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='SegRescale', scale_factor=0.125),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                    'gt_semantic_seg'
                ])
        ],
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing'),
        seg_prefix='/opt/ml/input/data/')

cfg.checkpoint_config = dict(max_keep_ckpts=2, interval=1)


cfg.work_dir = 'work_dirs/swin2_lr_test'
cfg.resume_from = '/opt/ml/SwinModel/work_dirs/swin2_1team/latest.pth'

cfg.optimizer = dict(
    type='AdamW',
    lr=0.000001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

# cfg.lr_config = dict(policy='CosineAnnealing',warmup='linear',warmup_iters=10,
#                     warmup_ratio=0.0001, min_lr_ratio=1e-7)

cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[100, 120])

print(f'Config:\n{cfg.pretty_text}')

model = build_detector(cfg.model)

datasets = [build_dataset(cfg.data.train)]

train_detector(model, datasets[0], cfg, distributed=False, validate=True)