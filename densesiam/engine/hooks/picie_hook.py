import os.path as osp
import time

import mmcv
import numpy as np
import torch
from mmcv.runner import HOOKS, Hook
from PIL import Image
from densesiam.utils import get_root_logger


try:
    import faiss
except ImportError:
    faiss = None


def get_datetime(time_delta):
    days_delta = time_delta // (24 * 3600)
    time_delta = time_delta % (24 * 3600)
    hour_delta = time_delta // 3600
    time_delta = time_delta % 3600
    mins_delta = time_delta // 60
    time_delta = time_delta % 60
    secs_delta = time_delta

    return f'{days_delta}:{hour_delta}:{mins_delta}:{secs_delta}'


@HOOKS.register_module()
class PiCIEClusterHook(Hook):
    """Hook that perform clustering before each epoch to produce pseudo labels.

    This hook should be sure to be executed after DistSamplerHook() because it
    will run the dataloader multiple times to perform clustering and produce
    pseudo labels, which need the dataloader to bump data in the same orders as
    those during training in the following epoch.
    """

    def __init__(self,
                 balance_loss=True,
                 in_channels=256,
                 num_classes=27,
                 num_init_batches=20,
                 num_batches=1,
                 log_interval=100,
                 kmeans_n_iter=20,
                 seed=2,
                 max_points_per_centroid=10000000,
                 reset_data_random=False,
                 label_dir='.labels'):
        self.logger = get_root_logger()
        self.balance_loss = balance_loss
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_init_batches = num_init_batches
        self.num_batches = num_batches
        self.log_interval = log_interval
        self.label_dir = label_dir
        self.kmeans_n_iter = kmeans_n_iter
        self.seed = seed
        self.max_points_per_centroid = max_points_per_centroid
        self.reset_data_random = reset_data_random

    def before_run(self, runner):
        self.label_dir = osp.join(runner.work_dir, self.label_dir)
        mmcv.mkdir_or_exist(self.label_dir)

    def before_epoch(self, runner):
        # compute pseudo labels before each epoch
        # run mini-batch k-means for two views
        assert len(runner.train_dataloaders) == 1, \
            'More than one training data loader is not allowed for PiCIE'
        dataloader = runner.train_dataloaders[0]
        dataloader.dataset.mode = 'compute'
        dataloader.dataset.reshuffle()
        model = runner.model
        model.module.eqv_pipeline = dataloader.dataset.eqv_pipeline
        model.eval()

        self.logger.info('Start computing centroids')
        t1 = time.time()
        centroids1, loss1 = self.minibatch_kmeans(dataloader, model, view=1)
        centroids2, loss2 = self.minibatch_kmeans(dataloader, model, view=2)
        self.logger.info(
            f'Centroids ready in {get_datetime(int(time.time() - t1))}'
            f'. Loss: {loss1:.4f} | {loss2:.4f}')

        model.module.reset_classifier(centroids1, centroids2)

        t2 = time.time()
        weight1 = self.compute_labels(dataloader, model, centroids1, view=1)
        weight2 = self.compute_labels(dataloader, model, centroids2, view=2)
        self.logger.info(
            f'Cluster labels ready in {get_datetime(int(time.time() - t2))}')

        if self.balance_loss:
            model.module.cls_loss1.weight = weight1
            model.module.cls_loss2.weight = weight2
            self.logger.info('Loss weight has been set for balance')

        dataloader.dataset.mode = 'train'
        dataloader.dataset.labeldir = self.label_dir
        model.train()
        if hasattr(self, 'reset_data_random') and self.reset_data_random:
            dataloader.dataset.reset_pipeline_randomness()
            model.module.eqv_pipeline = dataloader.dataset.eqv_pipeline

    @torch.no_grad()
    def minibatch_kmeans(self, dataloader, model, view):
        kmeans_loss = AverageMeter()
        faiss_module = get_faiss_module(self.in_channels, model.device.index)
        data_count = np.zeros(self.num_classes)
        featslist = []
        num_batches = 0
        first_batch = True
        centroids = None
        dataloader.dataset.view = view
        data_length = len(dataloader)

        for i, data in enumerate(dataloader):
            feats = model(mode='extract', view=view, **data)
            feats = flatten_feats(feats).detach().cpu()

            if num_batches < self.num_init_batches:
                featslist.append(feats)
                num_batches += 1
                if (num_batches == self.num_init_batches
                        or num_batches == len(dataloader)):
                    if first_batch:
                        # Compute initial centroids.
                        # By doing so, we avoid empty cluster problem from '
                        # mini-batch K-Means.
                        featslist = torch.cat(featslist).cpu().numpy()
                        centroids = self.get_init_centroids(
                            featslist, faiss_module)
                        loss, clusters = faiss_module.search(featslist, 1)
                        kmeans_loss.update(loss.mean())
                        self.logger.info(
                            f'Initial k-means loss: {kmeans_loss.avg:.4f}')

                        # Compute counts for each cluster.
                        for k in np.unique(clusters):
                            data_count[k] += len(np.where(clusters == k)[0])
                        first_batch = False
                    else:
                        assert centroids is not None
                        b_feat = torch.cat(featslist)
                        faiss_module = update_centroids(
                            faiss_module, centroids)
                        loss, clusters = faiss_module.search(b_feat.numpy(), 1)
                        kmeans_loss.update(loss.mean())
                        # Update centroids.
                        for k in np.unique(clusters):
                            idx_k = np.where(clusters == k)[0]
                            data_count[k] += len(idx_k)
                            centroid_lr = len(idx_k) / (data_count[k] + 1e-6)
                            centroids[k] = (1 - centroid_lr) * centroids[k] \
                                + centroid_lr * b_feat[idx_k].mean(
                                    0).numpy()

                    # Empty.
                    featslist = []
                    num_batches = self.num_init_batches - self.num_batches

                if (i % self.log_interval) == 0:
                    self.logger.info(
                        f'Saving features of view: {view}: [{i}/{data_length}]'
                        f', K-Means Loss: {kmeans_loss.avg:.4f}')

        centroids = torch.tensor(centroids, device=model.device)
        del faiss_module
        return centroids, kmeans_loss.avg

    @torch.no_grad()
    def compute_labels(self, dataloader, model, centroids, view):
        counts = torch.zeros(self.num_classes).to(model.device)
        dataloader.dataset.view = view
        save_dir = osp.join(self.label_dir, f'label_{view}')
        mmcv.mkdir_or_exist(save_dir)
        data_length = len(dataloader)

        for i, data in enumerate(dataloader):
            inds = data['idx']
            feats = model(mode='extract', view=view, **data)
            if view == 1:
                classifier = model.module.classifier1
            elif view == 2:
                classifier = model.module.classifier2
            else:
                raise ValueError(f'Unsupported view: {view}')

            scores = compute_dist(feats, centroids, classifier)
            preds = scores.argmax(dim=1)
            # Count for re-weighting.
            counts += torch.bincount(
                preds.flatten(), minlength=self.num_classes).float()

            preds = preds.detach().cpu().numpy().astype('uint8')
            for idx, idx_img in enumerate(inds):
                Image.fromarray(preds[idx]).save(
                    osp.join(save_dir, f'{idx_img}.png'))

            if (i % self.log_interval) == 0:
                self.logger.info(
                    f'Assigning labels of view: {view} [{i}/{data_length}]')

        weight = counts / counts.sum()
        return weight

    def get_init_centroids(self, featlist, index):
        assert self.in_channels == featlist.shape[-1]
        cluster = faiss.Clustering(self.in_channels, self.num_classes)
        cluster.seed = np.random.randint(self.seed)
        cluster.niter = self.kmeans_n_iter
        cluster.max_points_per_centroid = self.max_points_per_centroid
        cluster.train(featlist, index)

        out = faiss.vector_float_to_array(cluster.centroids)

        return out.reshape(self.num_classes, self.in_channels)


def compute_dist(feats, centroids, metric_function):
    # negative l2 squared
    centroids = centroids.unsqueeze(-1).unsqueeze(-1)
    dist = -(1 - 2 * metric_function(feats) +
             centroids.pow(2).sum(dim=1).unsqueeze(0))
    return dist


def flatten_feats(feats):
    if len(feats.size()) == 2:
        # feature already flattened.
        return feats

    feats = feats.flatten(2).transpose(2, 1).reshape(-1, feats.size(1))

    return feats


def get_faiss_module(in_channels, device=0):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = device
    idx = faiss.GpuIndexFlatL2(res, in_channels, cfg)

    return idx


def update_centroids(index, centroids):
    index.reset()
    index.add(centroids)

    return index


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
