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

            # ── Stage 1: Mosaic (50 % prob) ─────────────────────────────────
            # Combines 4 recent images into one ~2×resize canvas.
            # Each object appears at ¼ the area → forces small-object learning.
            # img_scale / crop_size are overridden by worker.py per user setting.
            dict(type='mmdet.CachedMosaic',
                 img_scale=(512, 512),
                 random_pop=False,
                 pad_val=114.0,
                 prob=0.5),

            # ── Stage 2: Scale jitter + crop ────────────────────────────────
            # Randomly resize to 0.5–2.0× of the target, then crop to target.
            # Exposes the model to objects at varying absolute pixel sizes.
            dict(type='mmdet.RandomResize',
                 scale=[(256, 256), (1024, 1024)],
                 keep_ratio=True),
            # Pad to at least target size (needed when RandomResize < target).
            dict(type='mmdet.Pad', size=(512, 512), pad_val=114),
            dict(type='mmdet.RandomCrop',
                 crop_size=(512, 512),
                 allow_negative_crop=False),

            # ── Stage 3: Photometric distortion ─────────────────────────────
            dict(type='DicePhotoMetricDistortion'),
            dict(type='DiceGaussianNoise', prob=0.5),

            # ── Stage 4: Geometric augmentation ─────────────────────────────
            dict(type='mmdet.RandomFlip',
                 prob=0.5,
                 direction=['horizontal', 'vertical']),
            dict(type='DiceRandomRotate', prob=0.9),
            dict(type='DiceRandomShift', prob=0.9),

            # ── Stage 5: Instance-level augmentation ────────────────────────
            # Copy-Paste: pastes complete rotated instances from cached images.
            # Only cx/cy are translated; w, h, angle are preserved exactly.
            dict(type='DiceCopyPaste', prob=0.3, max_num_pasted=10),

            # Random erasing: reduced patches to protect small objects.
            dict(type='DiceRandomErasing', n_patches=8, ratio=(0.02, 0.06)),

            # This is for Debug Pipeline. uncomment temporarily to visualise.
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
