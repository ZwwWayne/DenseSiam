# Dense Siamese Network for Dense Unsupervised Learning

## Introduction

This is an official release of the paper **Dense Siamese Network for Dense Unsupervised Learning**.

> [**Dense Siamese Network for Dense Unsupervised Learning**](https://arxiv.org/abs/2203.11075),
> Wenwei Zhang, Jiangmiao Pang, Kai Chen, Chen Change Loy
> In: Proc. European Conference on Computer Vision (ECCV), 2022
> [[arXiv](https://arxiv.org/abs/2203.11075)][[project page](https://www.mmlab-ntu.com/project/densesiam/index.html)][[Bibetex](https://github.com/ZwwWayne/DenseSiam#citation)]

## Results

### Semantic segmentation on curated COCO stuff-thing dataset

The results of DenseSiam and their corresponding configs on unsupervised semantic segmentation task are shown as below.
We also re-implemented PiCIE based on the [official code release](https://github.com/janghyuncho/PiCIE).

| Backbone | Method | Lr Schd | mIoU | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| R-18  | PiCIE | 10e       | 14.4 |[config](configs/picie/picie_r18_fpn_10e_coco_curated.py) | [model]() &#124;  [log]() |
| R-18  | DenseSiam | 10e     | 16.4 |[config](configs/densesiam/densesiam_r18_fpn_aux_seg-rebalance_4x64_sgd-fix-10e_coco-curated.py) | [model]() &#124;  [log]() |

### Unsupervised representation learning

| Backbone | Method | Lr Schd | COCO Mask mAP| Config | Pre-train Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| R-50  | DenseSiam | 1x        | 34.0 |[config](configs/) | [model]() &#124;  [log]() |

## Installation

It requires the following OpenMMLab packages:

- MIM >= 0.1.5
- MMCV-full >= v1.3.14

```bash
pip install openmim mmdet mmsegmentation
mim install mmcv-full
```

## Usage

### Data preparation

- Download the [training set](http://images.cocodataset.org/zips/train2017.zip) and the [validdation set](http://images.cocodataset.org/zips/val2017.zip) of COCO dataset as well as the [stuffthing map](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip).
- Unzip these data and place them as the following structure
- The `curated` directory copies the data split for unsupervised segmentation from [PiCIE](https://github.com/janghyuncho/PiCIE).

```text
data/
├── curated
│   ├── train2017
│   │   ├── Coco164kFull_Stuff_Coarse_7.txt
│   ├── val2017
│   │   ├── Coco164kFull_Stuff_Coarse_7.txt
├── coco
│   ├── annotations
│   │   ├── train2017
│   │   │   ├── xxxxxxxxx.png
│   │   ├── val2017
│   │   │   ├── xxxxxxxxx.png
│   ├── train2017
│   │   ├── xxxxxxxxx.jpeg
│   ├── val2017
│   │   ├── xxxxxxxxx.jpeg
```

### Training and testing

For training and testing, you can directly use mim to train and test the model

```bash
# train instance/panoptic segmentation models
sh ./tools/slurm_train.sh $PARTITION $JOBNAME $CONFIG $WORK_DIR

# test semantic segmentation models
sh ./tools/slurm_test.sh $PARTITION $JOBNAME $CONFIG $CHECKPOINT --eval mIoU
```

- PARTITION: the slurm partition you are using
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- CONFIG: the config files under the directory `configs/`
- JOBNAME: the name of the job that are necessary for slurm

## Acknowledgement

This codebase is based on [MMCV](https://github.com/open-mmlab/mmcv) and it benefits a lot from [PiCIE](https://github.com/janghyuncho/PiCIE) [MMSelfSup](https://github.com/open-mmlab/mmselfsup), and [Detectron2](https://github.com/facebookresearch/detectron2).

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

```bibtex
@inproceedings{zhang2022densesiam,
author = {Wenwei Zhang and Jiangmiao Pang and Kai Chen and Chen Change Loy},
title = {Dense Siamese Network for Dense Unsupervised Learning},
year = {2022},
booktitle = {ECCV},
}
```
