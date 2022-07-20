import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from densesiam.datasets.builder import PIPELINES
from PIL import Image, ImageFilter
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    InterpolationMode = Image


class IndexCompose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, index, img):
        for t in self.transforms:
            img = t(index, img)
        return img

    def reset_randomness(self):
        for t in self.transforms:
            t.reset_randomness()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class ResizeCenterCrop(object):
    """Resize and center crop."""

    def __init__(self, res):
        self.res = res

    def __call__(self, idx, image):
        image = TF.resize(image, self.res, InterpolationMode.BILINEAR)
        w, h = image.size
        left = int(round((w - self.res) / 2.))
        top = int(round((h - self.res) / 2.))

        return TF.crop(image, top, left, self.res, self.res)

    def __repr__(self):
        s = self.__class__.__name__ + '(res={self.res})'
        return s


@PIPELINES.register_module()
class ReplayRandomResize(object):

    def __init__(self, rmin, rmax, N):
        self.rmin = rmin
        self.rmax = rmax
        self.N = N
        self.reslist = [random.randint(rmin, rmax) for _ in range(N)]

    def reset_randomness(self):
        self.reslist = [
            random.randint(self.rmin, self.rmax) for _ in range(self.N)
        ]

    def __call__(self, index, image):
        return TF.resize(image, self.reslist[index],
                         InterpolationMode.BILINEAR)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'rmin={self.rmin}, '
        s += f'rmax={self.rmax}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomCrop(object):

    def __init__(self, res, N):
        self.res = res
        self.N = N
        self.cons = [(np.random.uniform(0, 1), np.random.uniform(0, 1))
                     for _ in range(N)]

    def reset_randomness(self):
        self.cons = [(np.random.uniform(0, 1), np.random.uniform(0, 1))
                     for _ in range(self.N)]

    def __call__(self, index, image):
        ws, hs = self.cons[index]
        w, h = image.size
        left = int(round((w - self.res) * ws))
        top = int(round((h - self.res) * hs))

        return TF.crop(image, top, left, self.res, self.res)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'res={self.res}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomHorizontalFlip(object):

    def __init__(self, N, p=0.5):
        self.p_ref = p
        self.N = N
        self.plist = np.random.random_sample(N)

    def reset_randomness(self):
        self.plist = np.random.random_sample(self.N)

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.hflip(image)
        else:
            return image

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'p={self.p_ref}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomGaussianBlur(object):

    def __init__(self, sigma, p, N):
        self.min_x = sigma[0]
        self.max_x = sigma[1]
        self.del_p = 1 - p
        self.p_ref = p
        self.N = N
        self.plist = np.random.random_sample(N)

    def reset_randomness(self):
        self.plist = np.random.random_sample(self.N)

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            x = self.plist[index] - self.p_ref
            m = (self.max_x - self.min_x) / self.del_p
            b = self.min_x
            s = m * x + b

            return image.filter(ImageFilter.GaussianBlur(radius=s))
        else:
            return image

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'min_x={self.min_x}, '
        s += f'max_x={self.max_x}, '
        s += f'p={self.p_ref}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomGrayScale(object):

    def __init__(self, p, N):
        self.grayscale = transforms.RandomGrayscale(
            p=1.)  # Deterministic (We still want flexible out_dim).
        self.p_ref = p
        self.N = N
        self.plist = np.random.random_sample(N)

    def reset_randomness(self):
        self.plist = np.random.random_sample(self.N)

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return self.grayscale(image)
        else:
            return image

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'p={self.p_ref}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomColorBrightness(object):

    def __init__(self, x, p, N):
        self.min_x = max(0, 1 - x)
        self.max_x = 1 + x
        self.p_ref = p
        self.N = N
        self.plist = np.random.random_sample(N)
        self.rlist = [random.uniform(self.min_x, self.max_x) for _ in range(N)]

    def reset_randomness(self):
        self.plist = np.random.random_sample(self.N)
        self.rlist = [
            random.uniform(self.min_x, self.max_x) for _ in range(self.N)
        ]

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.adjust_brightness(image, self.rlist[index])
        else:
            return image

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'min_x={self.min_x}, '
        s += f'max_x={self.max_x}, '
        s += f'p={self.p_ref}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomColorContrast(object):

    def __init__(self, x, p, N):
        self.min_x = max(0, 1 - x)
        self.max_x = 1 + x
        self.p_ref = p
        self.N = N
        self.plist = np.random.random_sample(N)
        self.rlist = [random.uniform(self.min_x, self.max_x) for _ in range(N)]

    def reset_randomness(self):
        self.plist = np.random.random_sample(self.N)
        self.rlist = [
            random.uniform(self.min_x, self.max_x) for _ in range(self.N)
        ]

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.adjust_contrast(image, self.rlist[index])
        else:
            return image

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'min_x={self.min_x}, '
        s += f'max_x={self.max_x}, '
        s += f'p={self.p_ref}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomColorSaturation(object):

    def __init__(self, x, p, N):
        self.min_x = max(0, 1 - x)
        self.max_x = 1 + x
        self.p_ref = p
        self.N = N
        self.plist = np.random.random_sample(N)
        self.rlist = [random.uniform(self.min_x, self.max_x) for _ in range(N)]

    def reset_randomness(self):
        self.plist = np.random.random_sample(self.N)
        self.rlist = [
            random.uniform(self.min_x, self.max_x) for _ in range(self.N)
        ]

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.adjust_saturation(image, self.rlist[index])
        else:
            return image

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'min_x={self.min_x}, '
        s += f'max_x={self.max_x}, '
        s += f'p={self.p_ref}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomColorHue(object):

    def __init__(self, x, p, N):
        self.min_x = -x
        self.max_x = x
        self.p_ref = p
        self.N = N
        self.plist = np.random.random_sample(N)
        self.rlist = [random.uniform(self.min_x, self.max_x) for _ in range(N)]

    def reset_randomness(self):
        self.plist = np.random.random_sample(self.N)
        self.rlist = [
            random.uniform(self.min_x, self.max_x) for _ in range(self.N)
        ]

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.adjust_hue(image, self.rlist[index])
        else:
            return image

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'min_x={self.min_x}, '
        s += f'max_x={self.max_x}, '
        s += f'p={self.p_ref}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomVerticalFlip(object):

    def __init__(self, N, p=0.5):
        self.p_ref = p
        self.N = N
        self.plist = np.random.random_sample(N)

    def reset_randomness(self):
        self.plist = np.random.random_sample(self.N)

    def __call__(self, indice, image):
        I = np.nonzero(self.plist[indice] < self.p_ref)[0]

        if len(image.size()) == 3:
            image_t = image[I].flip([1])
        else:
            image_t = image[I].flip([2])

        return torch.stack([
            image_t[np.where(I == i)[0][0]] if i in I else image[i]
            for i in range(image.size(0))
        ])

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'p={self.p_ref}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomHorizontalTensorFlip(object):

    def __init__(self, N, p=0.5):
        self.p_ref = p
        self.N = N
        self.plist = np.random.random_sample(N)

    def reset_randomness(self):
        self.plist = np.random.random_sample(self.N)

    def __call__(self, indice, image, is_label=False):
        I = np.nonzero(self.plist[indice] < self.p_ref)[0]

        if len(image.size()) == 3:
            image_t = image[I].flip([2])
        else:
            image_t = image[I].flip([3])

        return torch.stack([
            image_t[np.where(I == i)[0][0]] if i in I else image[i]
            for i in range(image.size(0))
        ])

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'p={self.p_ref}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayRandomResizedCrop(object):

    def __init__(self, N, res, scale=(0.5, 1.0)):
        self.res = res
        self.scale = scale
        self.N = N
        self.rscale = [np.random.uniform(*scale) for _ in range(N)]
        self.rcrop = [(np.random.uniform(0, 1), np.random.uniform(0, 1))
                      for _ in range(N)]

    def reset_randomness(self):
        self.rscale = [np.random.uniform(*self.scale) for _ in range(self.N)]
        self.rcrop = [(np.random.uniform(0, 1), np.random.uniform(0, 1))
                      for _ in range(self.N)]

    def random_crop(self, idx, img):
        ws, hs = self.rcrop[idx]
        res1 = int(img.size(-1))
        res2 = int(self.rscale[idx] * res1)
        i1 = int(round((res1 - res2) * ws))
        j1 = int(round((res1 - res2) * hs))

        return img[:, :, i1:i1 + res2, j1:j1 + res2]

    def __call__(self, indice, image):
        new_image = []
        res_tar = self.res // 4 if image.size(
            1) > 5 else self.res  # View 1 or View 2?

        for i, idx in enumerate(indice):
            img = image[[i]]
            img = self.random_crop(idx, img)
            img = F.interpolate(
                img, res_tar, mode='bilinear', align_corners=False)

            new_image.append(img)

        new_image = torch.cat(new_image)

        return new_image

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'res={self.res}, '
        s += f'scale={self.scale}, '
        s += f'N={self.N})'
        return s


@PIPELINES.register_module()
class ReplayParallelRandomResizedCrop(object):

    def __init__(self, N, res, scale=(0.5, 1.0)):
        self.res = res
        self.scale = scale
        self.N = N
        self.rscale = [np.random.uniform(*scale) for _ in range(N)]
        self.rcrop = [(np.random.uniform(0, 1), np.random.uniform(0, 1))
                      for _ in range(N)]

    def reset_randomness(self):
        self.rscale = [np.random.uniform(*self.scale) for _ in range(self.N)]
        self.rcrop = [(np.random.uniform(0, 1), np.random.uniform(0, 1))
                      for _ in range(self.N)]

    def random_crop(self, idx, img):
        ws, hs = self.rcrop[idx]
        res1 = int(img.size(-1))
        res2 = int(self.rscale[idx] * res1)
        i1 = int(round((res1 - res2) * ws))
        j1 = int(round((res1 - res2) * hs))

        return img[:, :, i1:i1 + res2, j1:j1 + res2]

    def __call__(self, indice, image):
        new_image = []
        res_tar = self.res // 4 if image.size(
            1) > 5 else self.res  # View 1 or View 2?

        for i, idx in enumerate(indice):
            img = image[[i]]
            img = self.random_crop(idx, img)
            img = F.interpolate(
                img, res_tar, mode='bilinear', align_corners=False)

            new_image.append(img)

        new_image = torch.cat(new_image)

        return new_image

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'res={self.res}, '
        s += f'scale={self.scale}, '
        s += f'N={self.N})'
        return s
