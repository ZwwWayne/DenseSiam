# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class DistAugSeed(Hook):
    """Data-loading sampler for distributed training.

    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """

    def before_epoch(self, runner):
        if hasattr(runner.model.module, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.data_loader.sampler.set_epoch(runner.epoch)
        elif hasattr(runner.data_loader.batch_sampler.sampler, 'set_epoch'):
            # batch sampler in pytorch warps the sampler as its attributes.
            runner.data_loader.batch_sampler.sampler.set_epoch(runner.epoch)
