# MuSE-LAMP

MuSE-LAMP provides image analysis code for four-channel, label-free digital LAMP experiments. It processes bright-field videos to identify droplets, classify their amplification status, and estimate target concentrations for each channel.

## Overview

The analysis pipeline combines:

- **YOLOv11 instance segmentation** to detect droplets and measure their projected areas.
- **BoT-SORT tracking** to associate droplets across frames and combine repeated observations.
- **ConvNeXt V2 classification** to distinguish positive and negative droplets.
- **Droplet volume correction and Poisson estimation** to calculate concentrations independently for each channel.

Target identity is assigned through the experimental channel layout, while the classifier determines droplet positivity.

## Main Files

| File | Purpose |
| --- | --- |
| `4C_workCode.py` | Main video analysis and quantification pipeline |
| `4C_workCode_debug.py` | Random-frame inspection and debugging |
| `botsort_reid.yaml` | Tracking configuration |
| `train_ConvNeXtV2-B_cls.py` | Binary classification model training |
| `train_seg_yolo.py`, `train_yolov11.py` | Segmentation dataset conversion and model training |
| `units/quantification.py` | Droplet volume and concentration calculations |

## Usage

Model weights are stored using [Git LFS](https://git-lfs.com/). After installing Git LFS and cloning this repository, download the weights with:

```bash
git lfs install
git lfs pull
```

The code uses Python with PyTorch, Ultralytics, timm, OpenCV, and supporting scientific computing libraries. Set up the required dependencies and update the video paths, model paths, and experiment-specific parameters in `4C_workCode.py`. Then run:

```bash
python 4C_workCode.py
```

Results include droplet tracking records and per-channel concentration summaries.

Only three local weight files referenced by the code are included:

| Weight | Purpose |
| --- | --- |
| `runs/segment/seg_runs/droplet_seg_v2/weights/best.pt` | Trained droplet segmentation model |
| `convnextv2b_cls_runs/ConvNeXtV2-B/best.pt` | Trained positive/negative droplet classifier |
| `yolo11n-seg.pt` | Pretrained initialization for segmentation training |

Classification training obtains its pretrained backbone through timm. Final-epoch checkpoints and unused legacy weights are excluded.

Experimental videos, datasets, standalone plotting scripts, and generated results are not included.
