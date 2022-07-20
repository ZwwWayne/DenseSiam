from mmcv.runner import HOOKS, Hook
from densesiam.utils import get_root_logger


@HOOKS.register_module()
class ReshuffleDatasetHook(Hook):

    def __init__(self, reset_data_random=False, reshuffle=True,):
        self.logger = get_root_logger()
        self.reset_data_random = reset_data_random
        self.reshuffle = reshuffle

    def before_epoch(self, runner):
        # compute pseudo labels before each epoch
        # run mini-batch k-means for two views
        assert len(runner.train_dataloaders) == 1, \
            'More than one training data loader is not allowed for PiCIE'
        dataloader = runner.train_dataloaders[0]
        if self.reshuffle:
            dataloader.dataset.reshuffle()
            self.logger.info('Dataset reshuffled')
        else:
            self.logger.info('Dataset not reshuffled')

        if hasattr(self, 'reset_data_random') and self.reset_data_random:
            dataloader.dataset.reset_pipeline_randomness()
        runner.model.module.eqv_pipeline = dataloader.dataset.eqv_pipeline
        self.logger.info(
            'Equivalence pipeline assigned to model after reshuffle')
