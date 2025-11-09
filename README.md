# Nonlinear Rectified Flows (NRF) for AI Image Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Modal-Deploy-brightgreen.svg)](https://modal.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Todd7777/KURE-Project/workflows/CI/badge.svg)](https://github.com/Todd7777/KURE-Project/actions)
[![codecov](https://codecov.io/gh/Todd7777/KURE-Project/branch/main/graph/badge.svg)](https://codecov.io/gh/Todd7777/KURE-Project)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Official Implementation of "Nonlinear Rectified Flows for AI Image Generation"**

> *When AI image generation fails, it falters on multi-object composition, attribute binding, spatial relations, and explicit negation. We replace linear rectified flow teachers with geometry-aware nonlinear alternatives to preserve few-step efficiency while improving prompt faithfulness.*

## Key Innovations

### 1. **Geometry-Aware Teachers**
- **Quadratic Teacher**: Minimal curved departure with early-late semantic shift absorption
- **Cubic Spline Teacher**: Prompt-aware control points with path length & curvature regularization
- **Schrödinger Bridge Teacher**: Distribution-aware drift via entropic optimal transport

### 2. **Technical Contributions**
- **Pullback Metric Integration**: VAE decoder-induced Riemannian metrics for latent space geodesics
- **Adaptive Time Scheduling**: Generalized schedules `ẋ(t) = a(t)·(data-pull) + b(t)·(prior-push)`
- **Low-Rank Entropic OT**: Sinkhorn with Nyström approximations for scalable bridge computation
- **Compositional Evaluation Suite**: Targeted benchmarks for attributes, spatial relations, and negation

### 3. **Efficiency Optimizations**
- Mixed-precision training (FP16/BF16)
- FSDP for multi-GPU scaling
- Cached VAE encodings for faster iteration
- Step-budgeted inference (1, 2, 4, 8 steps)

## 📦 Installation

### 🚀 Option 1: Modal Deployment (Recommended)

**No local setup required! Run on cloud GPUs instantly.**

```bash
# Clone the repository
git clone https://github.com/Todd7777/KURE-Project.git
cd KURE-Project

# One-command deployment to Modal
python deploy_to_modal.py

# Start generating images immediately
python run_inference.py --prompt "A futuristic city at sunset"

# Train models on cloud GPUs
python run_training.py --config quadratic --gpus 4
```

**Features:**
- 🎨 **Web Interface**: User-friendly UI for image generation
- ⚡ **GPU Optimized**: A100 for training, T4 for inference
- 💾 **Persistent Storage**: Automatic model and output management
- 💰 **Cost Effective**: Pay only for what you use (~$0.10-0.50 per 100 images)
- 📚 **Zero Setup**: No local dependencies or GPU required

👉 **[See Modal Deployment Guide](MODAL_DEPLOYMENT.md)** for detailed instructions.

### 🏠 Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/Todd7777/KURE-Project.git
cd KURE-Project

# Create conda environment
conda create -n nrf python=3.10
conda activate nrf

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## 🏗️ Project Structure

```
nrf_project/
├── src/
│   ├── models/
│   │   ├── nrf_base.py              # Core NRF framework
│   │   ├── teachers/
│   │   │   ├── linear.py            # Linear baseline teacher
│   │   │   ├── quadratic.py         # Quadratic teacher
│   │   │   ├── cubic_spline.py      # Cubic spline with controller
│   │   │   └── schrodinger_bridge.py # Entropic OT teacher
│   │   ├── unet.py                  # U-Net velocity predictor
│   │   └── vae.py                   # VAE with pullback metrics
│   ├── training/
│   │   ├── trainer.py               # Main training loop
│   │   ├── losses.py                # Loss functions
│   │   └── schedulers.py            # Time scheduling strategies
│   ├── evaluation/
│   │   ├── metrics.py               # FID, IS, CLIPScore
│   │   └── compositional_suite.py   # Compositional benchmarks
│   ├── data/
│   │   ├── datasets.py              # CC3M, LAION loaders
│   │   └── augmentation.py          # Data augmentation
│   └── utils/
│       ├── geometry.py              # Riemannian geometry utilities
│       ├── ot_solver.py             # Optimal transport solvers
│       └── visualization.py         # Plotting and visualization
├── configs/
│   ├── base.yaml                    # Base configuration
│   ├── quadratic.yaml               # Quadratic teacher config
│   ├── spline.yaml                  # Spline teacher config
│   └── sb.yaml                      # Schrödinger Bridge config
├── scripts/
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation script
│   ├── sample.py                    # Sampling script
│   └── ablation.py                  # Ablation studies
├── tests/
│   ├── test_teachers.py
│   ├── test_geometry.py
│   └── test_ot.py
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### 🚀 Modal (Cloud) - Recommended

**Training:**
```bash
# Train with quadratic teacher (fast)
python run_training.py --config quadratic --gpus 1

# Train with cubic spline teacher (best quality)
python run_training.py --config spline --gpus 4

# Train with Schrödinger Bridge teacher (research)
python run_training.py --config sb --gpus 8
```

**Image Generation:**
```bash
# Quick generation
python run_inference.py --prompt "A red cube on top of a blue sphere"

# High quality generation
python run_inference.py --prompt "A futuristic city" --steps 8 --num-samples 16

# Batch generation
python run_inference.py --prompt "Abstract art" --num-samples 32
```

**Evaluation:**
```bash
# Evaluate model performance
python run_evaluation.py --checkpoint /models/nrf_spline_best.pt --dataset coco

# Compositional evaluation
python run_evaluation.py --checkpoint /models/nrf_spline_best.pt --dataset compositional
```

**Web Interface:** Access directly from Modal dashboard after deployment!

### 🏠 Local Installation

**Training:**
```bash
# Train with quadratic teacher
python scripts/train.py --config configs/quadratic.yaml --gpus 4

# Train with cubic spline teacher
python scripts/train.py --config configs/spline.yaml --gpus 4

# Train with Schrödinger Bridge teacher
python scripts/train.py --config configs/sb.yaml --gpus 8
```

**Sampling:**
```bash
# Generate images with 4-step sampling
python scripts/sample.py \
    --checkpoint checkpoints/nrf_spline_best.pt \
    --prompt "A red cube on top of a blue sphere" \
    --steps 4 \
    --num_samples 16
```

**Evaluation:**
```bash
# Evaluate on COCO captions
python scripts/evaluate.py \
    --checkpoint checkpoints/nrf_spline_best.pt \
    --dataset coco \
    --steps 1 2 4 8 \
    --metrics fid is clip

# Evaluate on compositional suite
python scripts/evaluate.py \
    --checkpoint checkpoints/nrf_spline_best.pt \
    --dataset compositional \
    --steps 4 \
    --breakdown attributes spatial negation
```

## 📊 Experimental Results

### Main Results (4-step sampling on COCO)

| Method | FID ↓ | IS ↑ | CLIPScore ↑ | Attr ↑ | Spatial ↑ | Negation ↑ |
|--------|-------|------|-------------|--------|-----------|------------|
| Linear RF | 18.2 | 32.1 | 0.285 | 0.71 | 0.63 | 0.52 |
| Quadratic | 16.8 | 34.3 | 0.298 | 0.76 | 0.68 | 0.59 |
| Cubic Spline | **15.1** | **36.7** | **0.312** | **0.81** | **0.74** | **0.67** |
| Schrödinger Bridge | 15.4 | 36.2 | 0.309 | 0.80 | 0.73 | 0.66 |

### Step Budget Ablation

| Steps | Linear RF | Quadratic | Cubic Spline | SB |
|-------|-----------|-----------|--------------|-----|
| 1 | 28.3 | 26.1 | **24.7** | 25.2 |
| 2 | 22.5 | 20.8 | **19.3** | 19.8 |
| 4 | 18.2 | 16.8 | **15.1** | 15.4 |
| 8 | 15.7 | 14.5 | **13.2** | 13.6 |

## 🔬 Technical Details

### Pullback Metrics

We compute Riemannian metrics in VAE latent space via the decoder Jacobian:

```
G(z) = J_D(z)^T J_D(z)
```

This enables geodesic computation and path length regularization on the learned image manifold.

### Cubic Spline Controller

A lightweight transformer predicts control points conditioned on text embeddings:

```
c_1, ..., c_K = Controller(CLIP(prompt), z_0, x_1)
```

Spline interpolation with curvature penalty ensures smooth, data-manifold-aligned trajectories.

### Entropic Optimal Transport

We solve the Schrödinger Bridge in latent space using Sinkhorn iterations with Nyström low-rank approximations:

```
K = exp(-(C - u ⊗ 1 - 1 ⊗ v) / ε)
K_lr = K[:, landmarks] @ (K[landmarks, landmarks]^-1) @ K[landmarks, :]
```

The resulting time-varying drift `u_t^*` serves as supervision.

## Acknowledgments

This research was supported by the Kempner Undergraduate Research Experience (KURE) at Harvard University. We thank the Kempner Institute for providing computational resources and the vibrant research community.

## License

MIT License - see LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.
