import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ..backbones import picie_resnet as resnet
from ..builder import ARCHITECTURES
from .base import BaseArchitecture


@ARCHITECTURES.register_module()
class PiCIE(BaseArchitecture):

    def __init__(self,
                 arch,
                 pretrained,
                 fpn_mfactor=1,
                 out_channels=128,
                 num_classes=27,
                 loss_within_weight=0.5,
                 loss_cross_weight=0.5):
        super(PiCIE, self).__init__()
        self.backbone = resnet.__dict__[arch](pretrained=pretrained)
        self.decoder = FPNDecoder(fpn_mfactor, out_channels)

        self.classifier1 = ParameterFreeConv2d(
            out_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.classifier1.weight.data.normal_(0, 0.01)
        self.classifier1.bias.data.zero_()
        self.classifier2 = ParameterFreeConv2d(
            out_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.classifier2.weight.data.normal_(0, 0.01)
        self.classifier2.bias.data.zero_()

        self.cls_loss1 = nn.CrossEntropyLoss()
        self.cls_loss2 = nn.CrossEntropyLoss()

        self.loss_within_weight = loss_within_weight
        self.loss_cross_weight = loss_cross_weight

    @torch.no_grad()
    def reset_classifier(self, weight1, weight2):
        self.classifier1.weight.data = weight1.unsqueeze(-1).unsqueeze(-1)
        self.classifier2.weight.data = weight2.unsqueeze(-1).unsqueeze(-1)

    def encode_feature(self, img):
        feats = self.backbone(img)
        feats = self.decoder(feats)
        feats = F.normalize(feats, dim=1, p=2)
        return feats

    def forward_test(self, img, **kwargs):
        feats = self.encode_feature(img)
        outs = self.classifier1(feats)
        return outs

    def forward_train(self, img, label, idx=None):
        img_v1 = img[0]
        img_v2 = img[1]
        label_v1 = label[0]
        label_v2 = label[1]

        img_v1 = self.eqv_pipeline(idx.cpu().numpy(), img_v1)
        feats_v1 = self.encode_feature(img_v1)
        seg_v11 = self.classifier1(feats_v1)
        seg_v12 = self.classifier2(feats_v1)

        feats_v2 = self.encode_feature(img_v2)
        # random parameters in eqv_pipeline are all in numpy
        feats_v2 = self.eqv_pipeline(idx.cpu().numpy(), feats_v2)
        seg_v22 = self.classifier2(feats_v2)
        seg_v21 = self.classifier1(feats_v2)

        loss11 = self.cls_loss1(seg_v11, label_v1)
        loss22 = self.cls_loss2(seg_v22, label_v2)
        loss_within = (loss11 + loss22) * 0.5 * self.loss_within_weight

        loss12 = self.cls_loss2(seg_v12, label_v2)
        loss21 = self.cls_loss1(seg_v21, label_v1)
        loss_cross = (loss12 + loss21) * 0.5 * self.loss_cross_weight

        loss = dict(loss_within=loss_within, loss_cross=loss_cross)
        return loss

    def forward(self, img, mode='train', idx=None, view=-1, **kwargs):
        if mode == 'train':
            return self.forward_train(img, idx=idx, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            img = img[0]
            if view == 1:
                img = self.eqv_pipeline(idx.cpu().numpy(), img)
            feats = self.encode_feature(img)
            if view == 2:
                feats = self.eqv_pipeline(idx.cpu().numpy(), feats)
            return feats
        else:
            raise Exception('No such mode: {}'.format(mode))

    def run_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['idx']))

        return outputs


class FPNDecoder(nn.Module):

    def __init__(self, mfactor=1, out_channels=128):
        super(FPNDecoder, self).__init__()

        self.layer4 = nn.Conv2d(
            512 * mfactor // 8,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.layer3 = nn.Conv2d(
            512 * mfactor // 4,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.layer2 = nn.Conv2d(
            512 * mfactor // 2,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.layer1 = nn.Conv2d(
            512 * mfactor, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        o4 = self.upsample_add(o3, self.layer4(x['res2']))

        return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=False) + y


class ParameterFreeConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = padding if isinstance(padding, str) else _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.register_buffer(
            'weight',
            torch.empty((out_channels, in_channels // groups, *kernel_size)))

        if bias:
            self.register_buffer('bias', torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def forward(self, input):
        out = F.conv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
