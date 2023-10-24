_base_ = ['./segformer_mit-b0_8xb2-160k_ade20k-512x512-joint-train.py']

# model settings
model = dict(
    pretrained='pretrain/mit_b4.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

data_root = '/home/tiger/ADEChallengeData2016'
data_root_syn = '/home/tiger/ADE20K_Synthetic'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

dataset_real_train = dict(
    type='RepeatDataset',
    times=11,
    dataset=dict(
        type='ADE20KDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline
    )
)

dataset_syn_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type='ADE20KDataset',
        data_root=data_root_syn,
        data_prefix=dict(
            img_path='images_resampled', seg_map_path='annotations_filtered_resampled'),
        pipeline=train_pipeline
    )
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[dataset_real_train, dataset_syn_train]))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=320000, val_interval=8000)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
        end=3000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=3000,
        end=320000,
        by_epoch=False)
]
