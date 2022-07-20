# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry

from .base import DatasetEvaluatorsWrapper

EVALUATORS = Registry('evaluators')


def build_evaluator(cfg, default_args=None):
    if isinstance(cfg, list):
        evaluators = [
            EVALUATORS.build(x, default_args=default_args) for x in cfg
        ]
    elif isinstance(cfg, dict):
        evaluators = EVALUATORS.build(cfg, default_args=default_args)
    else:
        raise ValueError(
            f'Expect cfg to be a dict or a list, obtain {type(cfg)}')

    evaluator = DatasetEvaluatorsWrapper(evaluators)
    return evaluator
