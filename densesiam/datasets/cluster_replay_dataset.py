import os.path as osp

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Compose
from densesiam.datasets.base import BaseDataset
from densesiam.datasets.builder import DATASETS, DATASOURCES, PIPELINES
from densesiam.datasets.pipelines import IndexCompose
from densesiam.utils import get_root_logger

try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    InterpolationMode = Image


@DATASETS.register_module(force=True)
class ClusterReplayDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward two views of the
    img at a time (MoCo, SimCLR)."""

    def __init__(self,
                 data_source,
                 inv_pipelines,
                 eqv_pipelines,
                 shared_pipelines,
                 out_pipeline,
                 mode,
                 prefetch=False,
                 return_label=True,
                 res1=320,
                 res2=640):
        assert not data_source.get('return_label', False)
        self.data_source = DATASOURCES.build(data_source)
        default_args = dict(N=self.data_source.get_length())
        self.reshuffle()
        self.shared_pipeline = IndexCompose(
            [PIPELINES.build(p) for p in shared_pipelines])
        self.inv_pipelines = [
            IndexCompose([
                PIPELINES.build(p, default_args=default_args)
                for p in inv_pipelines
            ]) for _ in range(2)
        ]
        self.eqv_pipeline = IndexCompose([
            PIPELINES.build(p, default_args=default_args)
            for p in eqv_pipelines
        ])
        self.out_pipeline = Compose([PIPELINES.build(p) for p in out_pipeline])
        self.return_label = return_label

        self.prefetch = prefetch
        self.res1 = res1
        self.res2 = res2
        self.mode = mode
        self.view = -1

        logger = get_root_logger()
        logger.info(f'{self.__class__.__name__} initialized:\n'
                    f'Shared initial Pipeline:\n{self.shared_pipeline}\n\n'
                    f'Invariant Pipelines:\n{self.inv_pipelines}\n\n'
                    f'Equivalent Pipeline:\n{self.eqv_pipeline}\n\n'
                    f'Output Pipeline: {self.out_pipeline}\n')

    def reshuffle(self):
        """Generate random floats for all img data to deterministically random
        transform.

        This is to use random sampling but have the same samples during
        clustering and training within the same epoch.
        """
        self.shuffled_indices = np.arange(self.data_source.get_length())
        np.random.shuffle(self.shuffled_indices)

    def reset_pipeline_randomness(self):
        for x in self.inv_pipelines:
            x.reset_randomness()
        self.eqv_pipeline.reset_randomness()
        logger = get_root_logger()
        logger.info('randomness reset for pipelines')

    def __getitem__(self, idx):
        idx = self.shuffled_indices[idx]
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            (f'The output from the data source must be an Img, got {type(img)}'
             '. Please ensure that the list file does not contain labels.')
        img = self.transform_img(idx, img)
        label = self.transform_label(idx)
        data = dict(idx=idx, img=img)
        if label[0] is not None:
            data.update(label=label)
        return data

    def transform_label(self, idx):
        if hasattr(self, 'return_label') and not self.return_label:
            return (None, )

        # TODO Equiv. transform.
        if self.mode == 'train':
            # assume labels are saved as torch tensor
            # This should be consistent with the PiCIEHook
            label1_path = osp.join(self.labeldir, 'label_1', f'{idx}.png')
            label2_path = osp.join(self.labeldir, 'label_2', f'{idx}.png')
            # should avoid memcache here because the value of labels
            # always change after each epoch
            label1 = Image.open(label1_path)
            label2 = Image.open(label2_path)
            label1 = np.array(label1)
            label2 = np.array(label2)

            label1 = torch.from_numpy(label1).long()
            label2 = torch.from_numpy(label2).long()

            return label1, label2

        elif self.mode == 'baseline_train':
            label1_path = osp.join(self.labeldir, 'label_1', f'{idx}.png')
            label1 = Image.open(label1_path)
            label1 = np.array(label1)
            label1 = torch.from_numpy(label1).long()

            return (label1, )

        return (None, )

    def transform_img(self, idx, img):
        # resize and center crop the img data
        img = self.shared_pipeline(idx, img)
        if self.mode == 'compute':
            if self.view == 1:
                img = self.inv_pipelines[0](idx, img)
                img = self.out_pipeline(img)
            elif self.view == 2:
                img = self.inv_pipelines[1](idx, img)
                img = TF.resize(img, self.res1, InterpolationMode.BILINEAR)
                img = self.out_pipeline(img)
            else:
                raise ValueError('View [{}] is an invalid option.'.format(
                    self.view))
            return (img, )
        elif 'train' in self.mode:
            # Invariance transform.
            img1 = self.inv_pipelines[0](idx, img)
            img1 = self.out_pipeline(img1)

            if self.mode == 'baseline_train':
                return (img1, )

            img2 = self.inv_pipelines[1](idx, img)
            img2 = TF.resize(img2, self.res1, InterpolationMode.BILINEAR)
            img2 = self.out_pipeline(img2)

            return (img1, img2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(
                self.mode))

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = ClusterReplayDataset(
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='train2017',
            seg_prefix='stuffthingmaps/train2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/train2017/Coco164kFull_Stuff_Coarse_7.txt',
            memcached=False,
            mclient_path=None,
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
        out_pipeline=[
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg)
        ],
        prefetch=False,
        mode='compute',
        res1=320,
        res2=640)

    dataset.view = 1
    data = dataset[0]
    img = data['img'][0]
    assert img.shape == (3, 640, 640)
    dataset.reset_pipeline_randomness()
