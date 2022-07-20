import os
import os.path as osp

from PIL import Image
from densesiam.datasets.builder import DATASOURCES
from densesiam.datasets.data_sources.image_list import ImageList
from densesiam.utils import get_root_logger


@DATASOURCES.register_module(force=True)
class CocoImageList(ImageList):

    def __init__(self,
                 root,
                 img_prefix,
                 seg_prefix='stuffthingmaps/train2017',
                 img_postfix='.jpg',
                 seg_postfix='.png',
                 list_file=None,
                 file_client_args=dict(backend='disk'),
                 return_label=False):
        self.root = root
        self.img_prefix = osp.join(root, img_prefix)
        self.seg_prefix = osp.join(root, seg_prefix)
        self.seg_postfix = seg_postfix
        self.img_postfix = img_postfix
        self.return_label = return_label

        logger = get_root_logger()
        if list_file is None:
            img_names = [
                x for x in os.listdir(osp.join(self.root, f'{img_prefix}'))
            ]
        else:
            img_names = [
                f'{id_.rstrip()}{img_postfix}'
                for id_ in list(open(list_file, 'r'))
            ]
            logger.info(f'Loading images with {list_file}')

        self.fns = [f'{self.img_prefix}/{x}' for x in img_names]
        self.labels = [
            f'{self.seg_prefix}/{x.replace(img_postfix, seg_postfix)}'
            for x in img_names
        ]
        logger.info(f'Loaded {len(self.fns)} images from {self.img_prefix}')
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def get_sample(self, idx):
        img = self.load_img(self.fns[idx])
        img = img.convert('RGB')
        if self.return_label:
            target = self.load_img(self.labels[idx])
            return img, target
        else:
            return img


if __name__ == '__main__':
    import mmcv
    coco_source = CocoImageList(
        root='./data/coco',
        img_prefix='train2017',
        seg_prefix='stuffthingmaps/train2017',
        img_postfix='.jpg',
        seg_postfix='.png',
        list_file='data/curated/train2017/Coco164kFull_Stuff_Coarse_7.txt',
        memcached=False,
        mclient_path=None,
        return_label=True)
    assert coco_source.get_length() == 49629
    mmcv.check_file_exist(coco_source.fns[0])
    mmcv.check_file_exist(coco_source.labels[0])
    assert coco_source.fns[0].split(
        '/')[-1][:-4] == coco_source.labels[0].split('/')[-1][:-4]
    img, target = coco_source.get_sample(0)
    assert img.size == target.size

    coco_source = CocoImageList(
        root='./data/coco',
        img_prefix='train2017',
        seg_prefix='stuffthingmaps/train2017',
        img_postfix='.jpg',
        seg_postfix='.png',
        list_file=None,
        memcached=False,
        mclient_path=None,
        return_label=True)
    assert coco_source.get_length() == 118287
    mmcv.check_file_exist(coco_source.fns[0])
    mmcv.check_file_exist(coco_source.labels[0])
    assert coco_source.fns[0].split(
        '/')[-1][:-4] == coco_source.labels[0].split('/')[-1][:-4]
    img, target = coco_source.get_sample(0)
    assert img.size == target.size

    coco_source = CocoImageList(
        root='./data/coco',
        img_prefix='val2017',
        seg_prefix='stuffthingmaps/val2017',
        img_postfix='.jpg',
        seg_postfix='.png',
        list_file='data/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt',
        memcached=False,
        mclient_path=None,
        return_label=True)
    assert coco_source.get_length() == 2175
    mmcv.check_file_exist(coco_source.fns[0])
    mmcv.check_file_exist(coco_source.labels[0])
    assert coco_source.fns[0].split(
        '/')[-1][:-4] == coco_source.labels[0].split('/')[-1][:-4]
    img, target = coco_source.get_sample(0)
    assert img.size == target.size
