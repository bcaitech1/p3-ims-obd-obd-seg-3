model = dict(
 # HTC 적용
    type='HybridTaskCascade',         
    pretrained=None,
    backbone=dict(
 # Swin-S 적용
        type='SwinTransformer',        
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
 # FPN 그대로 적용
        type='FPN',                    
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(          
 # rpn은 그대로 적용         
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8], # EDA를 통해 큰 물체가 많으면 값 더크게 작은 물체가 많으면 작게
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
 # roi_head HTC에 맞게 변경
    roi_head=dict(                        
        type='HybridTaskCascadeRoIHead',

 # 요녀석이 중요한데 segmentation 데이터를 가지고 추가적인 작업을 해주는 애
        semantic_roi_extractor=dict(  
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=24,
            ignore_label=255,
            loss_weight=0.2),

        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
 # 요녀석은 HTC기본에 있는bboxhead 말고 swin에 있는 bboxhead로
                type='Shared2FCBBoxHead', 
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
 # Mask R-CNN에 필요한 애
        mask_roi_extractor=dict(   
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
 # HTC에 필요한 mask head
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=11,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=11,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=11,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

# model = dict(
#     type='HybridTaskCascade',
#     pretrained=None,
#     backbone=dict(
#         type='SwinTransformer',
#         embed_dim=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         window_size=7,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.3,
#         ape=False,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         use_checkpoint=False),
#     neck=dict(
#         type='FPN',
#         in_channels=[128, 256, 512, 1024],
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8],
#             ratios=[0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[0.0, 0.0, 0.0, 0.0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(
#             type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
#     roi_head=dict(
#         type='HybridTaskCascadeRoIHead',
#         semantic_roi_extractor=dict(  
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[8]),
#         semantic_head=dict(
#             type='FusedSemanticHead',
#             num_ins=5,
#             fusion_level=1,
#             num_convs=4,
#             in_channels=256,
#             conv_out_channels=256,
#             num_classes=24,
#             ignore_label=255,
#             loss_weight=0.2),

#         interleaved=True,
#         mask_info_flow=True,
#         num_stages=3,
#         stage_loss_weights=[1, 0.5, 0.25],

#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=[
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0.0, 0.0, 0.0, 0.0],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0.0, 0.0, 0.0, 0.0],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='ConvFCBBoxHead',
#                 num_shared_convs=4,
#                 num_shared_fcs=1,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0.0, 0.0, 0.0, 0.0],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=False,
#                 reg_decoded_bbox=True,
#                 norm_cfg=dict(type='BN', requires_grad=True),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
#         ],
#         mask_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         mask_head=dict(
#             type='FCNMaskHead',
#             num_convs=4,
#             in_channels=256,
#             conv_out_channels=256,
#             num_classes=11,
#             loss_mask=dict(
#                 type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
#     train_cfg=dict(
#         rpn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.7,
#                 neg_iou_thr=0.3,
#                 min_pos_iou=0.3,
#                 match_low_quality=True,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=256,
#                 pos_fraction=0.5,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=False),
#             allowed_border=0,
#             pos_weight=-1,
#             debug=False),
#         rpn_proposal=dict(
#             nms_across_levels=False,
#             nms_pre=2000,
#             nms_post=2000,
#             max_per_img=2000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=[
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.5,
#                     neg_iou_thr=0.5,
#                     min_pos_iou=0.5,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.6,
#                     neg_iou_thr=0.6,
#                     min_pos_iou=0.6,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.7,
#                     neg_iou_thr=0.7,
#                     min_pos_iou=0.7,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False)
#         ]),
#     test_cfg=dict(
#         rpn=dict(
#             nms_across_levels=False,
#             nms_pre=1000,
#             nms_post=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=dict(
#             score_thr=0.00,
#             nms=dict(type='nms', iou_threshold=0.4),
#             max_per_img=100,
#             mask_thr_binary=0.5)))



########################### 우리 모델 시작 ########################
# model = dict(
#  # HTC 적용
#     type='HybridTaskCascade',         
#     pretrained=None,
#     backbone=dict(
#  # Swin-S 적용
#         type='SwinTransformer',        
#         embed_dim=96,
#         depths=[2, 2, 18, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         mlp_ratio=4.,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.2,
#         ape=False,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         use_checkpoint=False),
#     neck=dict(
#  # FPN 그대로 적용
#         type='FPN',                    
#         in_channels=[96, 192, 384, 768],
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(          
#  # rpn은 그대로 적용         
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8], # EDA를 통해 큰 물체가 많으면 값 더크게 작은 물체가 많으면 작게
#             ratios=[0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
#  # roi_head HTC에 맞게 변경
#     roi_head=dict(                        
#         type='HybridTaskCascadeRoIHead',

#  # 요녀석이 중요한데 segmentation 데이터를 가지고 추가적인 작업을 해주는 애
#         semantic_roi_extractor=dict(  
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[8]),
#         semantic_head=dict(
#             type='FusedSemanticHead',
#             num_ins=5,
#             fusion_level=1,
#             num_convs=4,
#             in_channels=256,
#             conv_out_channels=256,
#             num_classes=11,
#             ignore_label=255,
#             loss_weight=0.2),

#         interleaved=True,
#         mask_info_flow=True,
#         num_stages=3,
#         stage_loss_weights=[1, 0.5, 0.25],
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=[
#             dict(
#  # 요녀석은 HTC기본에 있는bboxhead 말고 swin에 있는 bboxhead로
#                 type='Shared2FCBBoxHead', 
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
#         ],
#  # Mask R-CNN에 필요한 애
#         mask_roi_extractor=dict(   
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#  # HTC에 필요한 mask head
#         mask_head=[
#             dict(
#                 type='HTCMaskHead',
#                 with_conv_res=False,
#                 num_convs=4,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 num_classes=11,
#                 loss_mask=dict(
#                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
#             dict(
#                 type='HTCMaskHead',
#                 num_convs=4,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 num_classes=11,
#                 loss_mask=dict(
#                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
#             dict(
#                 type='HTCMaskHead',
#                 num_convs=4,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 num_classes=11,
#                 loss_mask=dict(
#                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
#         ]),
#     train_cfg=dict(
#         rpn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.7,
#                 neg_iou_thr=0.3,
#                 min_pos_iou=0.3,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=256,
#                 pos_fraction=0.5,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=False),
#             allowed_border=0,
#             pos_weight=-1,
#             debug=False),
#         rpn_proposal=dict(
#             nms_pre=2000,
#             max_per_img=2000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=[
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.5,
#                     neg_iou_thr=0.5,
#                     min_pos_iou=0.5,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.6,
#                     neg_iou_thr=0.6,
#                     min_pos_iou=0.6,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.7,
#                     neg_iou_thr=0.7,
#                     min_pos_iou=0.7,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False)
#         ]),
#     test_cfg=dict(
#         rpn=dict(
#             nms_pre=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=dict(
#             score_thr=0.001,
#             nms=dict(type='nms', iou_threshold=0.5),
#             max_per_img=100,
#             mask_thr_binary=0.5)))


dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.,
        hflag=False,
        aug_ratio=0.5),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True), 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                          (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                          (736, 1333), (768, 1333), (800, 1333)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type': 'Resize',
                      'img_scale': [(400, 1333), (500, 1333), (600, 1333)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (384, 600),
                      'allow_negative_crop': True
                  }, {
                      'type':
                      'Resize',
                      'img_scale': [(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
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
                    dict(type='MotionBlur', blur_limit=5, p=1.0),
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
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file='/opt/ml/input/data/train_all.json',
        img_prefix='/opt/ml/input/data/',
        seg_prefix='/opt/ml/input/data/',
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
                color_prob=0.,
                hflag=False,
                aug_ratio=0.5),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True), 
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type':
                    'Resize',
                    'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                  (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                  (736, 1333), (768, 1333), (800, 1333)],
                    'multiscale_mode':
                    'value',
                    'keep_ratio':
                    True
                }],
                          [{
                              'type': 'Resize',
                              'img_scale': [(400, 1333), (500, 1333), (600, 1333)],
                              'multiscale_mode': 'value',
                              'keep_ratio': True
                          }, {
                              'type': 'RandomCrop',
                              'crop_type': 'absolute_range',
                              'crop_size': (384, 600),
                              'allow_negative_crop': True
                          }, {
                              'type':
                              'Resize',
                              'img_scale': [(480, 1333), (512, 1333), (544, 1333),
                                            (576, 1333), (608, 1333), (640, 1333),
                                            (672, 1333), (704, 1333), (736, 1333),
                                            (768, 1333), (800, 1333)],
                              'multiscale_mode':
                              'value',
                              'override':
                              True,
                              'keep_ratio':
                              True
                          }]]),
            dict(
                type='Albu',
                # transforms=[
                #     dict(
                #         type='Cutout',
                #         num_holes=30,
                #         max_h_size=30,
                #         max_w_size=30,
                #         fill_value=[103.53, 116.28, 123.675],
                #         p=0.1),
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
                #             dict(type='RandomGamma'),
                #             dict(type='CLAHE')
                #         ],
                #         p=0.1),
                #     dict(
                #         type='JpegCompression',
                #         quality_lower=85,
                #         quality_upper=95,
                #         p=0.2),
                #     dict(type='ChannelShuffle', p=0.1),
                #     dict(
                #         type='OneOf',
                #         transforms=[
                #             dict(type='Blur', blur_limit=3, p=1.0),
                #             dict(type='MedianBlur', blur_limit=3, p=1.0),
                #             dict(type='MotionBlur', blur_limit=5, p=1.0),
                #             dict(type='GaussNoise'),
                #             dict(type='ImageCompression', quality_lower=75)
                #         ],
                #         p=0.1),
                #     dict(type='RandomRotate90', p=0.5)
                # ],
                transforms=[],
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
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
        ],
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')),
    val=dict(
        type='CocoDataset',
        ann_file='/opt/ml/input/data/val.json',
        img_prefix='/opt/ml/input/data/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')),
    test=dict(
        type='CocoDataset',
        ann_file='/opt/ml/input/data/test.json',
        img_prefix='/opt/ml/input/data/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')))

evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[27, 33])
# runner = dict(type='EpochBasedRunner', max_epochs=36)
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
checkpoint_config = dict(max_keep_ckpts=2, interval=1)
log_config = dict(
    interval=40,
    hooks=[dict(type='TensorboardLoggerHook'),
           dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'cascade_mask_rcnn_swin_base_patch4_window7.pth'
resume_from = None
workflow = [('train', 1)]
albu_train_transforms = [
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
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
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
]
seed = 2020
gpu_ids = [0]
work_dir = './work_dirs/swin2'



# #################################### Transfrom 복사
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# # augmentation strategy
# albu_train_transforms = [
#     dict(
#         type='VerticalFlip',
#         p=0.5),
#     dict(
#         type='RandomBrightnessContrast',
#         brightness_limit=[-0.1, 0.1],
#         contrast_limit=[-0.1, 0.1],
#         p=0.7),
#     dict(
#         type='CLAHE',
#         clip_limit=2.0),
#     dict(
#         type='HueSaturationValue',
#         hue_shift_limit=10,
#         sat_shift_limit=15,
#         val_shift_limit=10),
#     dict(
#         type='RandomRotate90'),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='MedianBlur', blur_limit=5),
#             dict(type='MotionBlur', blur_limit=5),
#             dict(type='GaussianBlur', blur_limit=5),
#         ],
#         p=0.7),
#     dict(
#         type='GaussNoise', var_limit=(5.0, 30.0), p=0.5)
# ]

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#  # augmentation에서 color건드려주니까 instaboost에서는 color 빼주기
#     dict(
#         type='InstaBoost',
#         action_candidate=('normal', 'horizontal', 'skip'),
#         action_prob=(1, 0, 0),
#         scale=(0.8, 1.2),
#         dx=15,
#         dy=15,
#         theta=(-1, 1),
#         color_prob=0.,
#         hflag=False,
#         aug_ratio=0.5),
#  # segmentation 적용을 위해 with_seg=True
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True), 
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='AutoAugment',
#          policies=[
#              [
#                  dict(type='Resize',
#                       img_scale=[(512, 512),(576, 576),(640, 640),(704, 704),(768, 768),
#                       (832, 832),(896, 896),(960, 960),(1024, 1024)],
#                       multiscale_mode='value',
#                       keep_ratio=True)
#              ],
#              [
#                  dict(type='Resize',
#                       img_scale=[(512, 512), (768, 768), (1024, 1024)],
#                       multiscale_mode='value',
#                       keep_ratio=True),
#                  dict(type='RandomCrop',
#                       crop_type='absolute_range',
#                       crop_size=(512, 512),
#                       allow_negative_crop=True),
#                  dict(type='Resize',
#                       img_scale=[(512, 512),(576, 576),(640, 640),(704, 704),(768, 768),
#                       (832, 832),(896, 896),(960, 960),(1024, 1024)],
#                       multiscale_mode='value',
#                       override=True,
#                       keep_ratio=True)
#              ]
#          ]),
#     dict(type='Pad', size_divisor=32),
#  # Albumetation은 Pad뒤에 둬야 오류 안뜨고 Normalize앞에 쓰는게 국룰
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
#         skip_img_without_anno=True),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='SegRescale', scale_factor=1 / 8),
#     dict(type='DefaultFormatBundle'),
#  # segmentation 적용을 위해 gt_semantic_seg 넣어주어야 함
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
# ]
# data = dict(train=dict(pipeline=train_pipeline))



# _base_ = [
#     '../_base_/datasets/coco_instance.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]

# model = dict(
#  # HTC 적용
#     type='HybridTaskCascade',         
#     pretrained=None,
#     backbone=dict(
#  # Swin-S 적용
#         type='SwinTransformer',        
#         embed_dim=96,
#         depths=[2, 2, 18, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         mlp_ratio=4.,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.2,
#         ape=False,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         use_checkpoint=False),
#     neck=dict(
#  # FPN 그대로 적용
#         type='FPN',                    
#         in_channels=[96, 192, 384, 768],
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(          
#  # rpn은 그대로 적용         
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8], # EDA를 통해 큰 물체가 많으면 값 더크게 작은 물체가 많으면 작게
#             ratios=[0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
#  # roi_head HTC에 맞게 변경
#     roi_head=dict(                        
#         type='HybridTaskCascadeRoIHead',

#  # 요녀석이 중요한데 segmentation 데이터를 가지고 추가적인 작업을 해주는 애
#         semantic_roi_extractor=dict(  
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[8]),
#         semantic_head=dict(
#             type='FusedSemanticHead',
#             num_ins=5,
#             fusion_level=1,
#             num_convs=4,
#             in_channels=256,
#             conv_out_channels=256,
#             num_classes=24,
#             ignore_label=255,
#             loss_weight=0.2),

#         interleaved=True,
#         mask_info_flow=True,
#         num_stages=3,
#         stage_loss_weights=[1, 0.5, 0.25],
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=[
#             dict(
#  # 요녀석은 HTC기본에 있는bboxhead 말고 swin에 있는 bboxhead로
#                 type='Shared2FCBBoxHead', 
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=11,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
#         ],
#  # Mask R-CNN에 필요한 애
#         mask_roi_extractor=dict(   
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#  # HTC에 필요한 mask head
#         mask_head=[
#             dict(
#                 type='HTCMaskHead',
#                 with_conv_res=False,
#                 num_convs=4,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 num_classes=11,
#                 loss_mask=dict(
#                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
#             dict(
#                 type='HTCMaskHead',
#                 num_convs=4,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 num_classes=11,
#                 loss_mask=dict(
#                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
#             dict(
#                 type='HTCMaskHead',
#                 num_convs=4,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 num_classes=11,
#                 loss_mask=dict(
#                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
#         ]),
#     train_cfg=dict(
#         rpn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.7,
#                 neg_iou_thr=0.3,
#                 min_pos_iou=0.3,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=256,
#                 pos_fraction=0.5,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=False),
#             allowed_border=0,
#             pos_weight=-1,
#             debug=False),
#         rpn_proposal=dict(
#             nms_pre=2000,
#             max_per_img=2000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=[
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.5,
#                     neg_iou_thr=0.5,
#                     min_pos_iou=0.5,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.6,
#                     neg_iou_thr=0.6,
#                     min_pos_iou=0.6,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.7,
#                     neg_iou_thr=0.7,
#                     min_pos_iou=0.7,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 mask_size=28,
#                 pos_weight=-1,
#                 debug=False)
#         ]),
#     test_cfg=dict(
#         rpn=dict(
#             nms_pre=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=dict(
#             score_thr=0.001,
#             nms=dict(type='nms', iou_threshold=0.5),
#             max_per_img=100,
#             mask_thr_binary=0.5)))
    
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# # augmentation strategy
albu_train_transforms = [
    dict(
        type='VerticalFlip',
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[-0.1, 0.1],
        contrast_limit=[-0.1, 0.1],
        p=0.7),
    dict(
        type='CLAHE',
        clip_limit=2.0),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10),
    dict(
        type='RandomRotate90'),
    dict(
        type='OneOf',
        transforms=[
            dict(type='MedianBlur', blur_limit=5),
            dict(type='MotionBlur', blur_limit=5),
            dict(type='GaussianBlur', blur_limit=5),
        ],
        p=0.7),
    dict(
        type='GaussNoise', var_limit=(5.0, 30.0), p=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
 # augmentation에서 color건드려주니까 instaboost에서는 color 빼주기
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.,
        hflag=False,
        aug_ratio=0.5),
 # segmentation 적용을 위해 with_seg=True
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True), 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(512, 512),(576, 576),(640, 640),(704, 704),(768, 768),
                      (832, 832),(896, 896),(960, 960),(1024, 1024)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(512, 512), (768, 768), (1024, 1024)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(512, 512),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(512, 512),(576, 576),(640, 640),(704, 704),(768, 768),
                      (832, 832),(896, 896),(960, 960),(1024, 1024)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Pad', size_divisor=32),
 # Albumetation은 Pad뒤에 둬야 오류 안뜨고 Normalize앞에 쓰는게 국룰
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
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
 # segmentation 적용을 위해 gt_semantic_seg 넣어주어야 함
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
data = dict(train=dict(pipeline=train_pipeline))

# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))
#  # cosine anealing scheduler
# lr_config = dict(policy='CosineAnnealing',warmup='linear',warmup_iters=3000,
#                     warmup_ratio=0.0001, min_lr_ratio=1e-7)
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=60)

# # do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )



# load_from = "cascade_mask_rcnn_swin_small_patch4_window7.pth" # 얘는 pretrain 모델 가져오는 경로
# # resume_from = '/opt/ml/code/swin/work_dirs/version/epoch_57.pth' # 얘는 학습 조건 그대로 가져와서 이어 돌릴 때 쓰는 애