_base_ = ['../default_runtime.py']

model = dict(
    type='DenseSimSiam',
    arch='resnet18',
    pretrained=True,
    fpn_mfactor=1,
    out_channels=128,
    num_proj_convs=2,
    num_aux_proj_convs=2,
    num_aux_classes=128,
    loss_aux_weight='auto',
    loss_simsiam_weight=1.0,
    loss_seg_weight=1.0,
    rebalance_seg=True,
    loss_kernel_cross_weight=0.1,
    num_classes=27)

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
out_pipeline = [dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='ClusterReplayDataset',
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='train2017',
            seg_prefix='stuffthingmaps/train2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/train2017/Coco164kFull_Stuff_Coarse_7.txt',
            file_client_args={{_base_.file_client_args}},
            return_label=False),
        inv_pipelines=[
            dict(type='ReplayRandomColorBrightness', x=0.3, p=0.8),
            dict(type='ReplayRandomColorContrast', x=0.3, p=0.8),
            dict(type='ReplayRandomColorSaturation', x=0.3, p=0.8),
            dict(type='ReplayRandomColorHue', x=0.1, p=0.8),
            dict(type='ReplayRandomGrayScale', p=0.2),
            dict(type='ReplayRandomGaussianBlur', sigma=[.1, 2.], p=0.5)
        ],
        eqv_pipelines=[
            dict(type='ReplayRandomResizedCrop', res=320, scale=(0.5, 1)),
            dict(type='ReplayRandomHorizontalTensorFlip')
        ],
        shared_pipelines=[dict(type='ResizeCenterCrop', res=640)],
        out_pipeline=out_pipeline,
        prefetch=False,
        return_label=False,
        mode='train',
        res1=320,
        res2=640),
    val=dict(
        type='CocoEvalDataset',
        samples_per_gpu=128,
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='val2017',
            seg_prefix='stuffthingmaps/val2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt',
            file_client_args={{_base_.file_client_args}},
            return_label=True),
        img_out_pipeline=out_pipeline,
        res=320))

# 0.05 * total_bs / 256; total_bs=num_gpus*samples_per_gpu
optimizer = dict(type='SGD', lr=0.05, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(grad_clip=None)
log_config = dict(interval=20)
# learning policy
lr_config = dict(policy='Fixed')
runner = dict(type='EpochBasedRunner', max_epochs=10)
evaluator = [
    dict(
        type='ClusterIoUEvaluator',
        distributed=True,
        num_classes=27,
        num_thing_classes=12,
        num_stuff_classes=15)
]

custom_hooks = [
    dict(type='ValidateHook', initial=True, interval=1, trial=-1),
    dict(type='ReshuffleDatasetHook'),
    dict(
        type='LossWeightStepUpdateHook',
        interval=1,
        steps=[8, 10],
        gammas=[0, 1.0],
        key_names=['loss_kernel_cross_weight'])
]

custom_imports = dict(
    imports=[
        'densesiam.models.architectures.densesiam',
        'densesiam.engine.hooks.validate_hook',
        'densesiam.engine.hooks.reshuffle_hook',
        'densesiam.engine.hooks.loss_weight_updater',
    ],
    allow_failed_imports=False)
