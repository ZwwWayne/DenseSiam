# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class MultiStepAlternationHook(Hook):
    """Data-loading sampler for distributed training.

    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """

    def __init__(self,
                 projector_interval=194,
                 backbone_interval=194,
                 neck_with_backbone=False,
                 neck_with_head=False):
        self.projector_train_iters = 0
        self.backbone_train_iters = 0
        self.projector_interval = projector_interval
        self.backbone_interval = backbone_interval
        self.neck_with_backbone = neck_with_backbone
        self.neck_with_head = neck_with_head
        assert not (neck_with_head == neck_with_backbone)

    def freeze_module(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def unfreeze_module(self, module):
        module.train()
        for param in module.parameters():
            param.requires_grad_(True)

    def freeze_backbone(self, model):
        self.freeze_module(model.backbone)
        if self.neck_with_backbone:
            self.freeze_module(model.decoder)
        self.backbone_freezed = True

    def unfreeze_backbone(self, model):
        self.unfreeze_module(model.backbone)
        if self.neck_with_backbone:
            self.unfreeze_module(model.decoder)
        self.backbone_freezed = False

    def freeze_head(self, model):
        self.freeze_module(model.projector)
        self.freeze_module(model.predictor)
        model.backbone.layer4[1].conv1.weight.requires_grad
        model.decoder.layer1.weight.requires_grad
        model.predictor[-1].weight.requires_grad
        if hasattr(model, 'aux_predictor'):
            self.freeze_module(model.aux_projector)
            self.freeze_module(model.aux_predictor)
        if self.neck_with_head:
            self.freeze_module(model.decoder)
        self.head_freezed = True

    def unfreeze_head(self, model):
        self.unfreeze_module(model.projector)
        self.unfreeze_module(model.predictor)
        if hasattr(model, 'aux_predictor'):
            self.unfreeze_module(model.aux_projector)
            self.unfreeze_module(model.aux_predictor)
        if self.neck_with_head:
            self.unfreeze_module(model.decoder)
        self.head_freezed = False

    def before_run(self, runner):
        model = runner.model.module
        self.train_projector = True
        self.train_backbone = False

    def before_train_iter(self, runner):
        model = runner.model.module
        # import pdb; pdb.set_trace()
        # print(f'model.decoder: {model.decoder.layer4.weight.grad}')
        # print(f'model.backbone: {model.backbone.layer4[-1].conv1.weight.grad}')
        # print(f'model.backbone: {model.predictor[-1].weight.grad}')
        if self.train_projector:
            self.projector_train_iters += 1
            self.unfreeze_head(model)
            self.freeze_backbone(model)
            if self.projector_train_iters % self.projector_interval == 0:
                self.train_projector = False
                self.train_backbone = True
        elif self.train_backbone:
            self.backbone_train_iters += 1
            self.unfreeze_backbone(model)
            self.freeze_head(model)
            if self.backbone_train_iters % self.backbone_interval == 0:
                self.train_projector = True
                self.train_backbone = False
        else:
            raise ValueError('Unexpected self.train_projector='
                             f'{self.train_projector}, and '
                             f'self.train_backbone={self.train_backbone}')
