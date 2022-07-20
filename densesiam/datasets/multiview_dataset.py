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
class MultiCropReplayDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward two views of the
    img at a time (MoCo, SimCLR)."""

    def __init__(self,
                 data_source,
                 inv_pipelines,
                 num_crops,
                 shared_pipelines,
                 out_pipeline,
                 mode,
                 prefetch=False,
                 return_label=True,
                 res1=320,
                 res2=640):
        assert not data_source.get('return_label', False)
        self.data_source = DATASOURCES.build(data_source)
        self.num_crops = num_crops
        self.shared_pipeline = IndexCompose(
            [PIPELINES.build(p) for p in shared_pipelines])

        self.inv_pipelines = [
            Compose([
                PIPELINES.build(p)
                for p in inv_pipelines
            ]) for _ in range(sum(num_crops))
        ]


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
                    f'Output Pipeline: {self.out_pipeline}\n')

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            (f'The output from the data source must be an Img, got {type(img)}'
             '. Please ensure that the list file does not contain labels.')
        img = self.transform_img(idx, img)
        data = dict(idx=idx, img=img)
        return data

    def transform_img(self, idx, img):
        # resize and center crop the img data
        img = self.shared_pipeline(idx, img)
        # Invariance transform.
        img1 = self.inv_pipelines[0](img)
        img1 = self.out_pipeline(img1)
        outs = (img1, )

        small_img = TF.resize(img, self.res1, InterpolationMode.BILINEAR)
        for i in range(sum(self.num_crops) - 1):
            new_img = self.inv_pipelines[i+1](small_img)
            new_img = self.out_pipeline(new_img)
            outs += (new_img, )

        return outs

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    file_client_args = dict(
        backend='petrel',
        path_mapping=dict({
            './data/': 's3://openmmlab/datasets/detection/',
            'data/': 's3://openmmlab/datasets/detection/'
    }))

    dataset = MultiCropReplayDataset(
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='train2017',
            seg_prefix='stuffthingmaps/train2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/train2017/Coco164kFull_Stuff_Coarse_7.txt',
            file_client_args=file_client_args,
            return_label=False),
        num_crops=(2, 2),
        inv_pipelines=[
            dict(
                type='RandomAppliedTrans',
                transforms=[
                    dict(
                        type='ColorJitter',
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1)
                ],
                p=0.8),
            dict(type='RandomGrayscale', p=0.2),
            dict(
                type='RandomAppliedTrans',
                transforms=[
                    dict(
                        type='GaussianBlur',
                        sigma_min=0.1,
                        sigma_max=2.0)
                ],
                p=0.5),
        ],
        shared_pipelines=[dict(type='ResizeCenterCrop', res=640)],
        out_pipeline=[
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg)
        ],
        prefetch=False,
        mode='train',
        res1=320,
        res2=640)

    dataset.view = 1
    data = dataset[0]
    assert len(data['img']) == 4
    assert data['img'][0].shape == (3, 640, 640)
    assert data['img'][1].shape == (3, 320, 320)
    assert data['img'][2].shape == (3, 320, 320)
    assert data['img'][3].shape == (3, 320, 320)
    assert not (data['img'][1] == data['img'][2]).all()
    assert not (data['img'][2] == data['img'][3]).all()
