# Wavelet-Guided Residual Diffusion for Enhanced Heavy Rain Image Restoration
[![DOI](https://zenodo.org/badge/1140577421.svg)](https://doi.org/10.5281/zenodo.18350102)

> **Note:** This repository contains the official PyTorch implementation of the manuscript **"Wavelet-Guided Residual Diffusion for Enhanced Heavy Rain Image Restoration"**, currently submitted to **The Journal of Supercomputing**. The code is provided to ensure the transparency and reproducibility of our experiments.

## 📖 Abstract

Outdoor vision systems are frequently impaired by heavy rain, which introduces directional streak blur and droplet-induced refraction, leading to uneven, orientation-dependent residuals. Traditional restoration methods often struggle with these heterogeneous degradations. We introduce **WGRD (Wavelet-Guided Residual Diffusion)**, a subband-aware residual diffusion framework designed specifically for heavy rain restoration. By analysing wavelet decompositions of heavy-rain images, we identify systematic energy concentration patterns within directional subbands, motivating a subband-factorized residual formulation. Our framework employs a wavelet-adaptive UNet with orientation-sensitive filtering and cross-scale fusion to coherently reconstruct streaks and refractive boundaries. A multi-domain objective function enforces structural coherence and spectral stability. Experiments on Rain100H, Rain100L, and RainDrop datasets demonstrate state-of-the-art performance with only **five sampling steps**, validating both the restoration effectiveness and computational efficiency of our method.

## 🚀 Key Features

* **Physics-Guided Formulation:** Incorporates subband energy weights to prioritize degradation-heavy components in the diffusion loss.
* **Wavelet-Adaptive Backbone:** A specialized UNet that processes low-frequency structure and high-frequency textures via separate, orientation-aware paths.
* **Efficient Inference:** Achieves high-quality restoration in just **5 sampling steps**.
* **Robust Training:** Built on PyTorch Lightning with custom heavy-rain augmentation strategies.

## 🛠️ Environment Setup

We recommend using Anaconda to manage the python environment.

```bash
# 1. Create a virtual environment
conda create -n wgrd python=3.8
conda activate wgrd

# 2. Install PyTorch (Please adjust CUDA version according to your GPU)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install required dependencies
pip install pytorch-lightning albumentations opencv-python matplotlib torchmetrics scipy

```

## 📂 Dataset Preparation

Please organize your datasets as follows. The `datamodule.py` is configured to read from this structure automatically.

```text
datasets/
├── Rain100H/
│   ├── train/
│   │   ├── rain/      
│   │   └── norain/    
│   └── test/
│       ├── rain/
│       └── norain/
└── RainDrop/
    ├── train/
    │   ├── data/      
    │   └── gt/        
    └── test_a/        
        ├── data/
        └── gt/

```

## ⚡ Training

To train the WGRD model, run `train.py`. The script supports automatic precision adjustment and distributed training.

### Train on Rain100H (Default)

```bash
python train.py \
    --dataset Rain100H \
    --data_dir datasets/Rain100H \
    --log_dir logs/wgrd_rain100h \
    --batch_size 4 \
    --denoising_model WGRD_WaveUnet \
    --mode residual \
    --lr_scheduler_type CosineAnnealingLR

```

### Train on RainDrop

```bash
python train.py \
    --dataset RainDrop \
    --data_dir datasets/RainDrop \
    --batch_size 4 \
    --log_dir logs/wgrd_raindrop \
    --denoising_model WGRD_WaveUnet

```

**Key Arguments:**

* `--denoising_model`: `WGRD_WaveUnet`.
* `--mode`: `residual`.
* `--noise_schedule`: `LinearPro` or `CosinePro`.
* `--pretrained_path`: Path to a checkpoint (.ckpt) for fine-tuning.

## 🔎 Inference / Testing

To evaluate the model on the test set, use `test.py`. This uses the efficient **5-step inference** logic described in the paper.

```bash
python test.py \
    --dataset Rain100H \
    --data_dir datasets/Rain100H \
    --log_dir results/test_rain100h \
    --model_ckpt logs/wgrd_rain100h/lightning_logs/version_0/checkpoints/best.ckpt

```

Results (PSNR/SSIM metrics and saved images) will be stored in `log_dir`.

## 📧 Contact

If you have any questions regarding the code or the paper, please feel free to open an GitHub issue.

```

```
