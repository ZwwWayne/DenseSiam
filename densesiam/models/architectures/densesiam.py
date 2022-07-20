import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import picie_resnet as resnet
from ..builder import ARCHITECTURES
from .base import BaseArchitecture
from .picie import FPNDecoder


def CESimilarity(p1, z2):
    # BCHW, BCHW -> BCHW
    loss = -F.softmax(z2, dim=1) * F.log_softmax(p1, dim=1)
    # BCHW -> BHW
    return loss.sum(dim=1)


def HardCESimilarity(p1, z2, rebalance=False):
    label = z2.argmax(dim=1)  # BCHW -> BHW
    num_classes = z2.size(1)
    if rebalance:
        counts = torch.bincount(label.flatten(), minlength=num_classes).float()
        weight = counts / counts.sum()  # unnecessary to recale by num_classes
        # BHW
        loss = F.cross_entropy(p1, label, weight=weight, reduction='none')
    else:
        # BHW
        loss = F.cross_entropy(p1, label, reduction='none')
    return loss


@ARCHITECTURES.register_module()
class DenseSimSiam(BaseArchitecture):

    def __init__(self,
                 arch,
                 pretrained,
                 fpn_mfactor=1,
                 out_channels=128,
                 hid_channels=128,
                 num_proj_convs=1,
                 num_global_convs=2,
                 num_aux_proj_convs=1,
                 num_aux_classes=-1,
                 num_kernel_proj_convs=2,
                 global_in_channels=512,
                 global_hid_channels=512,
                 global_out_channels=512,
                 test_with_pred=False,
                 loss_global_weight=-1,
                 loss_aux_weight='auto',
                 loss_simsiam_weight=1.0,
                 loss_seg_weight=0,
                 loss_kernel_cross_weight=0,
                 kernel_temp=0.1,
                 kernel_with_pred=True,
                 kernel_proj_final_bn=True,
                 kernel_within_p=True,
                 rebalance_seg=False,
                 num_classes=27):
        super(DenseSimSiam, self).__init__()
        self.backbone = resnet.__dict__[arch](pretrained=pretrained)
        self.decoder = FPNDecoder(fpn_mfactor, out_channels)
        self.test_with_pred = test_with_pred
        self.loss_global_weight = loss_global_weight
        self.loss_seg_weight = loss_seg_weight
        self.loss_simsiam_weight = loss_simsiam_weight
        self.rebalance_seg = rebalance_seg
        self.loss_kernel_cross_weight = loss_kernel_cross_weight
        self.kernel_with_pred = kernel_with_pred
        self.kernel_proj_final_bn = kernel_proj_final_bn
        self.kernel_within_p = kernel_within_p

        proj_convs = []
        for i in range(num_proj_convs):
            proj_convs.append(
                nn.Conv2d(out_channels, out_channels, 1, bias=False))
            proj_convs.append(nn.BatchNorm2d(out_channels))
            proj_convs.append(nn.ReLU(inplace=True))
        proj_convs.append(nn.Conv2d(hid_channels, num_classes, 1))
        proj_convs.append(nn.BatchNorm2d(num_classes, affine=False))
        self.projector = nn.Sequential(*proj_convs)

        self.predictor = nn.Sequential(
            nn.Conv2d(num_classes, hid_channels, 1, bias=False),
            nn.BatchNorm2d(hid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, num_classes, 1))

        self.num_aux_classes = num_aux_classes
        if num_aux_classes > 0:
            proj_convs = []
            for _ in range(num_aux_proj_convs):
                proj_convs.append(
                    nn.Conv2d(out_channels, out_channels, 1, bias=False))
                proj_convs.append(nn.BatchNorm2d(out_channels))
                proj_convs.append(nn.ReLU(inplace=True))
            proj_convs.append(nn.Conv2d(hid_channels, num_aux_classes, 1))
            proj_convs.append(nn.BatchNorm2d(num_aux_classes, affine=False))
            self.aux_projector = nn.Sequential(*proj_convs)

            self.aux_predictor = nn.Sequential(
                nn.Conv2d(num_aux_classes, hid_channels, 1, bias=False),
                nn.BatchNorm2d(hid_channels), nn.ReLU(inplace=True),
                nn.Conv2d(hid_channels, num_aux_classes, 1))

            self.loss_aux_weight = loss_aux_weight
            if self.loss_aux_weight == 'auto':
                log_aux = math.log(num_aux_classes)
                log_sim = math.log(num_classes)
                denominator = log_aux + log_sim
                self.loss_aux_weight = log_sim / denominator
                self.loss_simsiam_weight = log_aux / denominator

        if self.loss_global_weight > 0:
            proj_convs = []
            for i in range(num_global_convs):
                proj_convs.append(
                    nn.Linear(
                        global_in_channels, global_out_channels, bias=False))
                proj_convs.append(nn.BatchNorm1d(global_out_channels))
                proj_convs.append(nn.ReLU(inplace=True))
            proj_convs.append(
                nn.Linear(
                    global_out_channels, global_out_channels, bias=False))
            proj_convs.append(
                nn.BatchNorm1d(global_out_channels, affine=False))
            self.global_projector = nn.Sequential(*proj_convs)

            self.global_predictor = nn.Sequential(
                nn.Linear(
                    global_out_channels, global_hid_channels, bias=False),
                nn.BatchNorm1d(global_hid_channels), nn.ReLU(inplace=True),
                nn.Linear(global_hid_channels, global_out_channels))

        proj_convs = []
        for i in range(num_kernel_proj_convs):
            proj_convs.append(
                nn.Linear(out_channels, out_channels, bias=False))
            proj_convs.append(nn.BatchNorm1d(out_channels))
            proj_convs.append(nn.ReLU(inplace=True))
        proj_convs.append(nn.Linear(out_channels, out_channels, bias=False))
        if self.kernel_proj_final_bn:
            proj_convs.append(nn.BatchNorm1d(out_channels, affine=False))
        self.kernel_projector = nn.Sequential(*proj_convs)
        if self.kernel_with_pred:
            norm_layer = nn.BatchNorm1d(hid_channels)
            self.kernel_predictor = nn.Sequential(
                nn.Linear(out_channels, hid_channels, bias=False), norm_layer,
                nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))

        self.kernel_temp = kernel_temp
        self.criterion = CESimilarity
        self.loss_seg = HardCESimilarity
        self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def encode_feature(self, img, return_res5=False):
        feats = self.backbone(img)
        mid = feats
        feats = self.decoder(feats)
        if return_res5:
            return feats, mid
        else:
            return feats

    def forward_test(self, img, **kwargs):
        feats = self.encode_feature(img)
        outs = self.projector(feats)
        if hasattr(self, 'test_with_pred') and self.test_with_pred:
            outs = self.predictor(outs)
        return outs

    def kernel_contrast(self, feats_v1, feats_v2, z1, z2):
        losses = dict()
        if self.loss_kernel_cross_weight < 0:
            return losses

        num_classes = z1.size(1)
        mask1 = z1.softmax(dim=1)
        mask2 = z2.softmax(dim=1)

        ctr1 = torch.einsum('bnhw,bchw->bnc', mask1, feats_v1)
        ctr2 = torch.einsum('bnhw,bchw->bnc', mask2, feats_v2)

        ctr_z1 = self.kernel_projector(ctr1.view(-1, ctr1.size(-1)))
        ctr_z2 = self.kernel_projector(ctr2.view(-1, ctr2.size(-1)))
        if self.kernel_with_pred:
            ctr_p1 = self.kernel_predictor(ctr_z1).view_as(ctr1)
            ctr_p2 = self.kernel_predictor(ctr_z2).view_as(ctr2)
        else:
            ctr_p1 = ctr_z1.view_as(ctr1)
            ctr_p2 = ctr_z2.view_as(ctr2)
        ctr_z2 = ctr_z2.view_as(ctr2)
        ctr_z1 = ctr_z1.view_as(ctr1)

        if self.loss_kernel_cross_weight >= 0:

            gt = torch.eye(num_classes).to(ctr1.device).bool()
            ctr_p1 = F.normalize(ctr_p1, p=2, dim=-1)
            ctr_p2 = F.normalize(ctr_p2, p=2, dim=-1)
            ctr_z1 = F.normalize(ctr_z1, p=2, dim=-1)
            ctr_z2 = F.normalize(ctr_z2, p=2, dim=-1)
            mat12 = torch.einsum('bnc,bmc->bnm', ctr_p1,
                                 ctr_z2) / self.kernel_temp
            mat21 = torch.einsum('bnc,bmc->bnm', ctr_p2,
                                 ctr_z1) / self.kernel_temp
            loss_cross = -(mat12.log_softmax(dim=-1)[:, gt].mean() +
                           mat21.log_softmax(dim=-1)[:, gt].mean()) * 0.5

            losses.update(loss_kernel_cross=loss_cross *
                          self.loss_kernel_cross_weight)

        return losses

    def forward_train(self, img, idx=None):
        img_v1 = img[0]
        img_v2 = img[1]

        img_v1 = self.eqv_pipeline(idx.cpu().numpy(), img_v1)
        feats_v1, mid1 = self.encode_feature(img_v1, return_res5=True)

        z1 = self.projector(feats_v1)
        p1 = self.predictor(z1)

        feats_v2, mid2 = self.encode_feature(img_v2, return_res5=True)
        # random parameters in eqv_pipeline are all in numpy
        feats_v2 = self.eqv_pipeline(idx.cpu().numpy(), feats_v2)

        z2 = self.projector(feats_v2)
        p2 = self.predictor(z2)

        loss_simsiam = (self.criterion(p1, z2.detach()).mean() +
                        self.criterion(p2, z1.detach()).mean()) * 0.5

        loss = dict(loss_simsiam=loss_simsiam * self.loss_simsiam_weight)

        loss_kernels = self.kernel_contrast(feats_v1, feats_v2, z1, z2)
        loss.update(loss_kernels)

        if self.loss_seg_weight > 0:
            loss_seg = (self.loss_seg(
                z1, z1.detach(), self.rebalance_seg).mean() + self.loss_seg(
                    z2, z2.detach(), self.rebalance_seg).mean()) * 0.5
            loss.update(loss_seg=loss_seg * self.loss_seg_weight)

        if self.num_aux_classes > 0:
            aux_z1 = self.aux_projector(feats_v1)
            aux_p1 = self.aux_predictor(aux_z1)
            aux_z2 = self.aux_projector(feats_v2)
            aux_p2 = self.aux_predictor(aux_z2)
            loss_aux = (self.criterion(aux_p1, aux_z2.detach()).mean() +
                        self.criterion(aux_p2, aux_z1.detach()).mean()) * 0.5
            loss.update(loss_aux=loss_aux * self.loss_aux_weight)

        if self.loss_global_weight > 0:
            global_z1 = F.adaptive_avg_pool2d(mid1['res5'], (1, 1)).flatten(1)
            global_z2 = F.adaptive_avg_pool2d(mid2['res5'], (1, 1)).flatten(1)
            global_z1 = self.global_projector(global_z1)
            global_p1 = self.global_predictor(global_z1)
            global_z2 = self.global_projector(global_z2)
            global_p2 = self.global_predictor(global_z2)
            loss_global = (
                self.criterion(global_p1, global_z2.detach()).mean() +
                self.criterion(global_p2, global_z1.detach()).mean()) * 0.5

            loss.update(loss_global=loss_global * self.loss_global_weight)

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
