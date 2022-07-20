import io
import os

import mmcv
from PIL import Image

from ..builder import DATASOURCES
from .utils import McLoader


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff)


@DATASOURCES.register_module()
class ImageList(object):

    def __init__(self,
                 root,
                 list_file,
                 file_client_args=dict(backend='disk'),
                 return_label=True):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.has_labels = len(lines[0].split()) == 2
        self.return_label = return_label
        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split() for l in lines])
            self.labels = [int(l) for l in self.labels]
        else:
            # assert self.return_label is False
            self.fns = [l.strip() for l in lines]
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def get_length(self):
        return len(self.fns)

    def load_img(self, filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        img_bytes = self.file_client.get(filename)
        img = pil_loader(img_bytes)
        return img

    def get_sample(self, idx):
        img = self.load_img(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return img, target
        else:
            return img
