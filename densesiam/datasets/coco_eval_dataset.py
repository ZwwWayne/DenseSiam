import mmcv
import numpy as np
import torch
import torchvision.transforms.functional as TF
from densesiam.datasets.base import BaseDataset
from densesiam.datasets.builder import DATASETS, DATASOURCES, PIPELINES
from densesiam.utils import get_root_logger
from PIL import Image
from torchvision.transforms import Compose

try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    InterpolationMode = Image


@DATASETS.register_module()
class CocoEvalDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward two views of the
    img at a time (MoCo, SimCLR)."""

    def __init__(self,
                 data_source,
                 img_out_pipeline,
                 ann_fine2coarse='data/fine_to_coarse_dict.pickle',
                 mode='test',
                 thing=True,
                 stuff=True,
                 res=128):
        assert data_source.get('return_label', False)
        self.data_source = DATASOURCES.build(data_source)
        self.res = res
        self.stuff = stuff
        self.thing = thing
        self.ann_fine2coarse = ann_fine2coarse
        self.fine2coarse = self.get_fine2coarse(ann_fine2coarse)
        self.img_out_pipeline = Compose(
            [PIPELINES.build(p) for p in img_out_pipeline])
        logger = get_root_logger()
        logger.info(f'{self.__class__.__name__} initialized:\n'
                    f'Output Pipeline: {self.img_out_pipeline}\n')

    def __getitem__(self, idx):
        img, label = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image) \
            and isinstance(label, Image.Image), \
            ('The output from the data source must be an Img, got: '
             f'{type(img)}. Please ensure that the list file does '
             'not contain labels.')
        img, label = self.transform_data(img, label)
        return dict(idx=idx, img=img, label=label)

    def get_fine2coarse(self, ann_fine2coarse):
        """Map fine label indexing to coarse label indexing."""
        d = mmcv.load(ann_fine2coarse)
        fine_to_coarse_dict = d['fine_index_to_coarse_index']
        fine_to_coarse_dict[255] = -1
        return fine_to_coarse_dict

    def transform_data(self, image, label):
        # TODO: encapsulate it with data pipeline
        # TODO: avoid resizing labels and calculate mIoU at the original sizes
        # 1. Resize
        image = TF.resize(image, self.res, InterpolationMode.BILINEAR)
        label = TF.resize(label, self.res, InterpolationMode.NEAREST)

        # 2. CenterCrop
        w, h = image.size
        left = int(round((w - self.res) / 2.))
        top = int(round((h - self.res) / 2.))

        image = TF.crop(image, top, left, self.res, self.res)
        label = TF.crop(label, top, left, self.res, self.res)

        # 3. Transformation
        image = self.img_out_pipeline(image)
        label = self._label_transform(label)

        return image, label

    def _label_transform(self, label):
        """In COCO-Stuff, there are 91 Things and 91 Stuff. 91 Things (0-90)

        => 12 superclasses (0-11) 91 Stuff (91-181) => 15 superclasses (12-26)

        For [Stuff-15], which is the benchmark IIC uses, we only use 15 stuff
        superclasses.
        """
        # TODO: considering move it to the evaluator if possible
        label = np.array(label)
        fine2coarse = np.vectorize(lambda x: self.fine2coarse[x])
        label = fine2coarse(label)  # Map to superclass indexing.
        mask = label >= 255  # Exclude unlabelled.

        # Start from zero.
        if self.stuff and not self.thing:
            # This makes all Things categories negative (ignored.)
            label[mask] -= 12
        elif self.thing and not self.stuff:
            # This makes all Stuff categories negative (ignored.)
            mask = label > 11
            label[mask] = -1

        # Tensor-fy
        label = torch.LongTensor(label)
        return label

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CocoEvalDataset(
        data_source=dict(
            type='CocoImageList',
            root='./data/coco',
            img_prefix='val2017',
            seg_prefix='stuffthingmaps/val2017',
            img_postfix='.jpg',
            seg_postfix='.png',
            list_file='data/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt',
            memcached=False,
            mclient_path=None,
            return_label=True),
        img_out_pipeline=[
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg)
        ],
        res=128)

    data = dataset[0]
    img = data['img']
    assert img.shape == (3, 128, 128)
