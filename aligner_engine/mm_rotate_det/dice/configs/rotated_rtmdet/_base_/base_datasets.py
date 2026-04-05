backend_args = None

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=False,
    dataset=dict(
        type='DiceRotateDetDataset',
        no_rotation=False,
        ann_file='',
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
            dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
            # dict(type='DiceSharpen'),
            # dict(type='mmdet.CachedMosaic', img_scale=(512, 512)),
            dict(type='DicePhotoMetricDistortion'),
            dict(type='DiceGaussianNoise', prob=0.5),
            dict(
                type='mmdet.RandomFlip',
                prob=0.5,
                direction=['horizontal', 'vertical']),
            dict(type='DiceRandomRotate', prob=0.9),
            dict(type='DiceRandomShift', prob=0.9),
            dict(type='DiceRandomErasing', n_patches=20, ratio=(0.05, 0.1)),
            # This is for Debug Pipeline. remove this annotation temporarily when you want to debug
            # dict(type='DiceShowImage'),
            dict(type='mmdet.PackDetInputs')
        ]))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DiceRotateDetDataset',
        no_rotation=False,
        ann_file='',
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
            dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
            # avoid bboxes being resized
            dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = val_dataloader

val_evaluator = dict(type='DiceDOTAMetric', metric='mAP', iou_thrs=0.75)
test_evaluator = val_evaluator

trainval_split = dict(
    valid_ratio=0.3, keep_split=False, data_thr=10, max_val=300
)
