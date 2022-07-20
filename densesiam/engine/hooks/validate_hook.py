import time

import mmcv
import torch
from mmcv.runner import HOOKS, Hook, get_dist_info
from torch.utils.data import Dataset
from densesiam.utils.comm import get_rank


@HOOKS.register_module()
class ValidateHook(Hook):
    """Validation hook.
    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        dist_mode (bool): Use distributed evaluation or not. Default: True.
        initial (bool): Whether to evaluate before the training starts.
            Default: True.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, initial=True, interval=1, trial=-1):
        self.initial = initial
        self.interval = interval
        self.trial = trial

    def before_run(self, runner):
        if self.initial:
            self._run_validate(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        self._run_validate(runner)

    @torch.no_grad()
    def _run_validate(self, runner):
        runner.model.eval()
        runner.evaluator.reset()
        dataloader = runner.val_dataloader
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataloader.dataset))

        time.sleep(2)  # This line can prevent deadlock problem in some cases.
        for i, data in enumerate(dataloader):
            outputs = runner.model(mode='test', **data)
            runner.evaluator.process(data, outputs)
            if rank == 0:
                batch_size = len(outputs) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
            if self.trial != -1 and i > self.trial:
                break

        eval_results = runner.evaluator.evaluate()
        for name, val in eval_results.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        runner.model.train()
