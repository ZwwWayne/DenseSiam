import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop
from densesiam.models.builder import COMPONENTS
from densesiam.utils.comm import get_rank
import numpy as np


@COMPONENTS.register_module()
class ReplayRandomHorizontalFlip(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.params = None

    def reset(self):
        self.params = None

    def init_params(self, image):
        # curr_state = torch.random.get_rng_state()
        # curr_rank = get_rank()
        # import pdb; pdb.set_trace()
        # torch.random.manual_seed(curr_rank + curr_state + self.epoch)
        # this ensures the randomness across ranks
        self.params = torch.rand(image.size(0)) < self.p
        # avoid breaking the current random state
        # torch.random.set_rng_state(curr_state)

    def forward(self, image, *args, **kwargs):
        if self.params is None:
            self.init_params(image)

        image_t = image.clone()
        image_t[self.params] = TF.hflip(image[self.params])
        return image_t

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'p={self.p})'
        return s


@COMPONENTS.register_module()
class ReplayRandomResizedCrop(RandomResizedCrop):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = None

    def reset(self):
        self.params = None

    def get_params(self, img, scale, ratio):
        width, height = TF._get_image_size(img)
        i, j, h, w = super().get_params(img, scale, ratio)
        i, j, h, w = i / height, j / width, h / height, w / width
        return i, j, h, w, height, width

    def init_params(self, imgs):
        # curr_state = torch.random.get_rng_state()
        # curr_rank = get_rank()
        # import pdb; pdb.set_trace()
        # torch.random.set_rng_state(curr_rank + curr_state + self.epoch)
        # this ensures the randomness across ranks
        self.params = [
            self.get_params(img, self.scale, self.ratio)
            for img in imgs
        ]
        # avoid breaking the current random state
        # torch.random.set_rng_state(curr_state)

    def __call__(self, imgs, *args, size=None, **kwargs):
        if self.params is None:
            self.init_params(imgs)

        if size is None:
            size = self.size

        new_image = []
        for idx, img in enumerate(imgs):
            img = imgs[idx]
            i, j, h, w, height, width = self.params[idx]
            real_height, real_width = img.shape[-2:]
            i, j, h, w = int(i * real_height), int(j * real_width), \
                int(h * real_height), int(w * real_width)

            img = TF.resized_crop(
                img, i, j, h, w, size, self.interpolation)

            new_image.append(img)

        new_image = torch.stack(new_image, dim=0)
        return new_image


@COMPONENTS.register_module()
class ReplayRandomResizedKeepRatioCrop(RandomResizedCrop):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = None

    def reset(self):
        self.params = None

    def get_params(self, img, scale, ratio):
        rscale = np.random.uniform(*self.scale)
        rcrop = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
        res1 = int(img.size(-1))
        res2 = int(rscale * res1)
        width, height = TF._get_image_size(img)
        i = int(round((res1 - res2) * rcrop[0]))
        j = int(round((res1 - res2) * rcrop[1]))
        i, j, h, w = i / height, j / width,  res2 / height, res2 / width
        return i, j, h, w, height, width

    def init_params(self, imgs):
        # curr_state = torch.random.get_rng_state()
        # curr_rank = get_rank()
        # import pdb; pdb.set_trace()
        # torch.random.set_rng_state(curr_rank + curr_state + self.epoch)
        # this ensures the randomness across ranks
        self.params = [
            self.get_params(img, self.scale, self.ratio)
            for img in imgs
        ]
        # avoid breaking the current random state
        # torch.random.set_rng_state(curr_state)

    def __call__(self, imgs, *args, size=None, **kwargs):
        if self.params is None:
            self.init_params(imgs)

        if size is None:
            size = self.size

        new_image = []
        for idx, img in enumerate(imgs):
            img = imgs[idx]
            i, j, h, w, height, width = self.params[idx]
            real_height, real_width = img.shape[-2:]
            i, j, h, w = int(i * real_height), int(j * real_width), \
                int(h * real_height), int(w * real_width)

            img = TF.resized_crop(
                img, i, j, h, w, size, self.interpolation)

            new_image.append(img)

        new_image = torch.stack(new_image, dim=0)
        return new_image
