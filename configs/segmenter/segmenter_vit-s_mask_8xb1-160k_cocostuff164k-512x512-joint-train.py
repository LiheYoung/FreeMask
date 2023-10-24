_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/coco-stuff164k_joint_train.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth'  # noqa

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        img_size=(512, 512),
        embed_dims=384,
        num_heads=6,
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=384,
        channels=384,
        num_classes=171,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

# train_dataloader = dict(
#     # num_gpus: 8 -> batch_size: 8
#     batch_size=1)
val_dataloader = dict(batch_size=1)

data_root = '/home/tiger/COCO'
data_root_syn = '/home/tiger/COCO_Synthetic'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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
    times=4,
    dataset=dict(
        type='COCOStuffDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train2017', seg_map_path='annotations/train2017'),
        pipeline=train_pipeline
    )
)

dataset_syn_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type='COCOStuffDataset',
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
        end=160000,
        by_epoch=False)
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
