# RVO OCT Multi-Task Learning Framework

This repository contains a PyTorch-based pipeline for analyzing retinal vein occlusion (RVO) images using multiple related tasks:

- pathology segmentation
- anatomy segmentation
- hemorrhage focus (HF) prediction
- burden regression

The project uses a multi-task model implemented under the `models/` directory and dataset loaders under the `dataset/` directory.

## Project Overview

The workflow is organized around several training scripts:

- `train_multitask.py` - trains the combined multi-task model
- `train_pathology_segmentation.py` - trains a pathology segmentation model
- `train_anatomy_segmentation.py` - trains an anatomy segmentation model
- `train_hf_localization.py` - trains a hemorrhage focus localization model
- `train_burden_regression.py` - trains a burden regression model
- `train_unet_scratch.py` - trains a UNet from scratch
- `train_vgg16_unet.py` - trains a VGG16-based UNet

## Repository Structure

- `dataset/` - dataset definitions and preprocessing helpers
- `models/` - model architectures
- `processed/` - processed annotations, masks, and heatmaps
- `results/` - experiment outputs and generated artifacts

## Required Dependencies

Install the main Python dependencies with:

```bash
pip install torch torchvision opencv-python numpy pandas tqdm
```

## Data Layout

The project expects the following directories:

- `RVO-Lesion/Image_Seg/` - image and mask data
- `RVO-Lesion/RVO_Lesion_Labelme/` - LabelMe annotation files
- `processed/` - generated masks, heatmaps, and CSV files


## Training

To train the multi-task model, run:

```bash
python train_multitask.py
```

Other training scripts can be run similarly:

```bash
python train_pathology_segmentation.py
python train_anatomy_segmentation.py
python train_hf_localization.py
python train_burden_regression.py
```

## Outputs

Training scripts will save:

- model weights as `.pth` files
- checkpoints under `checkpoints/`
- processed outputs under `processed/`
- evaluation or visualization results under `results/`

## Notes

- The project uses CUDA when available and falls back to CPU automatically.
- Some scripts may require that the raw dataset folder is present in the repository root.

