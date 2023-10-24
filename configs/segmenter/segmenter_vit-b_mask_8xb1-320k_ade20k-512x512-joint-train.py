_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/ade20k_joint_train.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
# optimizer = dict(lr=0.001, weight_decay=0.0)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# train_dataloader = dict(
#     # num_gpus: 8 -> batch_size: 8
#     batch_size=1)
val_dataloader = dict(batch_size=1)

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
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[dataset_real_train, dataset_syn_train]))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=320000,
        by_epoch=False)
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=320000, val_interval=8000)
