from math import cos, pi

from densesiam.utils import get_root_logger
from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class LossWeightUpdateHook(Hook):

    def __init__(self,
                 warmup_ratio=0,
                 by_epoch=True,
                 policy='linear',
                 interval=1,
                 key_names=['loss_kernel_cross_weight']):

        self.by_epoch = by_epoch
        self.policy = policy
        self.key_names = key_names
        self.warmup_ratio = warmup_ratio
        self.interval = interval

    def get_regular_lw(self, runner):
        progress = runner.epoch
        max_progress = runner.max_epochs

        if self.policy == 'linear':
            k = (1 - progress / max_progress) * (1 - self.warmup_ratio)
            warmup_lw = [_lw * (1 - k) for _lw in self.base_lw]
        elif self.policy == 'exp':
            k = self.warmup_ratio**(1 - progress / max_progress)
            warmup_lw = [_lw * (1 - k) for _lw in self.base_lw]
        return warmup_lw

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        self.base_lw = [
            getattr(runner.model.module, key) for key in self.key_names
        ]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        logger = get_root_logger()
        regular_lw = self.get_regular_lw(runner)
        for key, lw in zip(self.key_names, regular_lw):
            logger.info(f'set {key} as {lw}')
            setattr(runner.model.module, key, lw)


@HOOKS.register_module()
class LossWeightStepUpdateHook(LossWeightUpdateHook):

    def __init__(self,
                 by_epoch=True,
                 interval=1,
                 steps=[9, 10],
                 gammas=[0, 1.0],
                 key_names=['loss_kernel_cross_weight']):

        self.steps = steps
        self.gammas = gammas
        self.by_epoch = by_epoch
        self.key_names = key_names
        self.interval = interval

    def get_regular_lw(self, runner):
        progress = runner.epoch
        for s, gamma in zip(self.steps, self.gammas):
            if progress < s:
                exp = gamma
                break
        warmup_lw = [_lw * exp for _lw in self.base_lw]
        return warmup_lw
