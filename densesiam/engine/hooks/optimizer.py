# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
from itertools import chain

from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.fp16_utils import LossScaler, wrap_fp16_model
from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version
from torch.nn.utils import clip_grad

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


@HOOKS.register_module(force=True)
class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()
