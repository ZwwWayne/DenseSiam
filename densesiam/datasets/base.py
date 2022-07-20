from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .builder import DATASOURCES, PIPELINES


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_source (dict): Data source defined in
            `densesiam.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `densesiam.datasets.pipelines`.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        self.data_source = DATASOURCES.build(data_source)
        pipeline = [PIPELINES.build(p) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch

    def __len__(self):
        return self.data_source.get_length()

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def evaluate(self, scores, keyword, logger=None, **kwargs):
        pass
