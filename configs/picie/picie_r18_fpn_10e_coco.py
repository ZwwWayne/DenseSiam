_base_ = ['../default_runtime.py']

model = dict(
    type='PiCIE',
    arch='resnet18',
    pretrained=True,
    fpn_mfactor=1,
    out_channels=128,
    num_classes=27)

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
out_pipeline = [dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)]

data = dict(
    samples_per_gpu=128,
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
            list_file=None,
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
        mode='compute',
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
            list_file=None,
            file_client_args={{_base_.file_client_args}},
            return_label=True),
        img_out_pipeline=out_pipeline,
        res=320))

optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
log_config = dict(interval=50)
# learning policy
lr_config = dict(policy='step', warmup=None, step=10)
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
    dict(
        type='PiCIEClusterHook',
        balance_loss=True,
        in_channels=128,
        num_classes=27,
        # official code uses 20 for batch size=256 in clustering
        # we use batch size 128 in all cases thus it becomes 40
        num_init_batches=40,
        num_batches=1,
        log_interval=100,
        kmeans_n_iter=20,
        seed=2,
        label_dir='.labels')
]

custom_imports = dict(
    imports=[
        'densesiam.models.architectures.picie',
        'densesiam.engine.hooks.validate_hook',
    ],
    allow_failed_imports=False)
