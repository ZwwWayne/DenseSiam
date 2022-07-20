import numpy as np
import torch
import torch.nn.functional as F
from densesiam.core.evaluators.builder import EVALUATORS
from densesiam.utils import get_root_logger
from densesiam.utils.comm import all_gather, is_main_process, synchronize
from mmcv.utils import print_log
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from terminaltables import AsciiTable

from .base import DatasetEvaluator


@EVALUATORS.register_module()
class ClusterIoUEvaluator(DatasetEvaluator):
    """This module assumes the pixels to be ignored are marked by -1 in the
    label."""

    def __init__(self,
                 distributed=True,
                 num_classes=27,
                 num_thing_classes=-1,
                 num_stuff_classes=-1):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self.num_classes = num_classes
        self._distributed = distributed
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.reset()

    def reset(self):
        logger = get_root_logger()
        logger.info(f'Reset {self.__class__.__name__}')
        self._conf_matrix = np.zeros((self.num_classes, self.num_classes))

    @torch.no_grad()
    def process(self, inputs, outputs):
        label = inputs['label']
        B, C, H, W = outputs.size()
        probs = F.interpolate(
            outputs, label.shape[-2:], mode='bilinear', align_corners=False)
        preds = probs.topk(1, dim=1)[1].view(B, -1).cpu().numpy()
        label = label.view(B, -1).cpu().numpy()
        self._conf_matrix += scores(label, preds, self.num_classes)

    def evaluate(self):
        logger = get_root_logger()
        logger.info(f'Start metric evaluation in {self.__class__.__name__}')
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
            logger.info('Confusion matrix synchronized')

        # m = linear_assignment(self._conf_matrix.max() - self._conf_matrix)
        match = linear_sum_assignment(self._conf_matrix.max() -
                                      self._conf_matrix)
        # Evaluate.
        new_hist = np.zeros((self.num_classes, self.num_classes))
        for idx in range(self.num_classes):
            # new_hist[m[idx, 1]] = self._conf_matrix[idx]
            new_hist[match[1][idx]] = self._conf_matrix[idx]

        # NOTE: Now [new_hist] is re-ordered to 12 thing + 15 stuff classses.
        total_res = get_result_metrics(new_hist)
        results = dict(
            aAcc=total_res['aAcc'],
            mIoU=total_res['mIoU'],
            mAcc=total_res['mAcc'])
        if self.num_thing_classes != -1:
            thing_res = get_result_metrics(
                new_hist[:self.num_thing_classes, :self.num_thing_classes])
            stuff_res = get_result_metrics(new_hist[self.num_thing_classes:,
                                                    self.num_thing_classes:])

            results.update(
                aAcc_th=thing_res['aAcc'],
                mIoU_th=thing_res['mIoU'],
                mAcc_th=thing_res['mAcc'],
                aAcc_st=stuff_res['aAcc'],
                mIoU_st=stuff_res['mIoU'],
                mAcc_st=stuff_res['mAcc'])

        headers = ['', 'aAcc', 'mIoU', 'mAcc']
        data = [headers]
        data.append([
            'All', f"{results['aAcc']:.4f}", f"{results['mIoU']:.4f}",
            f"{results['mAcc']:.4f}"
        ])
        data.append([
            'Things', f"{results['aAcc_th']:.4f}", f"{results['mIoU_th']:.4f}",
            f"{results['mAcc_th']:.4f}"
        ])
        data.append([
            'Stuff', f"{results['aAcc_st']:.4f}", f"{results['mIoU_st']:.4f}",
            f"{results['mAcc_st']:.4f}"
        ])
        table = AsciiTable(data)
        logger = get_root_logger()
        print_log(
            'Segmentation Evaluation Results:\n' + table.table, logger=logger)
        return results


def _fast_hist(label_true, label_pred, n_class):
    # Exclude unlabelled data.
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask] + label_pred[mask],
        minlength=n_class**2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist


def get_result_metrics(histogram):
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp

    iou = tp / (tp + fp + fn)
    acc = tp / (tp + fn)
    aAcc = np.sum(tp) / np.sum(histogram)

    result = {
        'IoU': iou,
        'mIoU': np.nanmean(iou),
        'Acc': acc,
        'mAcc': np.nanmean(acc),
        'aAcc': aAcc
    }

    result = {k: 100 * v for k, v in result.items()}

    return result
