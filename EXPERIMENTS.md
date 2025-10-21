# Experiment Guide for Nonlinear Rectified Flows

This guide provides step-by-step instructions for reproducing all experiments in the paper.

---

## Prerequisites

### 1. Environment Setup

```bash
# Create conda environment
conda create -n nrf python=3.10
conda activate nrf

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Data Preparation

#### Download Datasets

**CC3M (Conceptual Captions 3M)**:
```bash
# Download using img2dataset
img2dataset --url_list cc3m.tsv \
    --input_format "tsv" \
    --url_col "url" \
    --caption_col "caption" \
    --output_folder data/cc3m \
    --processes_count 16 \
    --thread_count 64 \
    --image_size 256
```

**LAION-Aesthetics**:
```bash
# Download high-quality subset
img2dataset --url_list laion_aesthetics_6plus.parquet \
    --input_format "parquet" \
    --output_folder data/laion \
    --processes_count 16 \
    --thread_count 64 \
    --image_size 256
```

**COCO Captions**:
```bash
# Download COCO 2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip -d data/coco/
unzip val2017.zip -d data/coco/
unzip annotations_trainval2017.zip -d data/coco/
```

#### Precompute Text Embeddings (Optional, for faster training)

```bash
python -c "
from src.data.datasets import precompute_text_embeddings
precompute_text_embeddings('cc3m', 'train', 'data/cc3m_train_embeddings.pt')
precompute_text_embeddings('cc3m', 'val', 'data/cc3m_val_embeddings.pt')
"
```

### 3. Pretrain VAE (if not using pretrained)

```bash
python scripts/train_vae.py \
    --dataset cc3m \
    --batch_size 128 \
    --epochs 50 \
    --lr 1e-4 \
    --output checkpoints/vae_cc3m.pt
```

---

## Main Experiments

### Experiment 1: Linear RF Baseline

Train the baseline linear rectified flow:

```bash
# Single GPU
python scripts/train.py --config configs/base.yaml --gpus 1

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/base.yaml \
    --gpus 4
```

**Expected results** (4 steps):
- FID: ~18.2
- IS: ~32.1
- CLIPScore: ~0.285

### Experiment 2: Quadratic Teacher

Train with quadratic teacher:

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/quadratic.yaml \
    --gpus 4
```

**Variants**:

**Fixed alpha**:
```yaml
# configs/quadratic.yaml
teacher:
  alpha: 0.5
  learnable: false
```

**Learnable alpha**:
```yaml
teacher:
  alpha: 0.5
  learnable: true
```

**Adaptive alpha** (context-dependent):
```yaml
teacher:
  adaptive: true
  alpha_min: -0.5
  alpha_max: 1.0
```

**Expected results** (4 steps):
- FID: ~16.8
- IS: ~34.3
- CLIPScore: ~0.298

### Experiment 3: Cubic Spline Teacher

Train with cubic spline teacher:

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/spline.yaml \
    --gpus 4
```

**Hyperparameter sweep** (control points):
```bash
for K in 1 3 5; do
    python scripts/train.py \
        --config configs/spline.yaml \
        --teacher.num_control_points $K \
        --training.run_name "nrf_spline_K${K}"
done
```

**Expected results** (4 steps, K=3):
- FID: ~15.1
- IS: ~36.7
- CLIPScore: ~0.312

### Experiment 4: Schrödinger Bridge Teacher

Train with Schrödinger Bridge teacher:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/sb.yaml \
    --gpus 8
```

**Note**: SB requires more GPU memory due to OT computation. Use 8 GPUs or reduce batch size.

**Expected results** (4 steps):
- FID: ~15.4
- IS: ~36.2
- CLIPScore: ~0.309

---

## Evaluation

### Standard Metrics (FID, IS, CLIPScore)

Evaluate on COCO validation set:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/nrf_spline_best.pt \
    --dataset coco \
    --eval_type standard \
    --steps "1 2 4 8" \
    --num_samples 5000 \
    --output_dir results/
```

### Compositional Evaluation

Generate compositional prompts:

```bash
python -c "
from src.evaluation.compositional_suite import CompositionalPromptGenerator
generator = CompositionalPromptGenerator()
generator.save_suite('data/compositional_suite.json')
"
```

Evaluate on compositional suite:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/nrf_spline_best.pt \
    --eval_type compositional \
    --compositional_suite data/compositional_suite.json \
    --compositional_steps 4 \
    --output_dir results/
```

### Step Budget Ablation

Compare all methods at different step budgets:

```bash
for steps in 1 2 4 8; do
    for method in linear quadratic spline sb; do
        python scripts/evaluate.py \
            --checkpoint checkpoints/nrf_${method}_best.pt \
            --dataset coco \
            --steps $steps \
            --output_dir results/${method}_${steps}steps/
    done
done
```

---

## Ablation Studies

### Ablation 1: Spline Control Points

```bash
for K in 1 2 3 4 5; do
    python scripts/train.py \
        --config configs/spline.yaml \
        --teacher.num_control_points $K \
        --training.run_name "ablation_K${K}"
done
```

### Ablation 2: Regularization Weights

```bash
for lambda1 in 0.0 0.05 0.1 0.2 0.5; do
    for lambda2 in 0.0 0.025 0.05 0.1 0.2; do
        python scripts/train.py \
            --config configs/spline.yaml \
            --teacher.path_length_weight $lambda1 \
            --teacher.curvature_weight $lambda2 \
            --training.run_name "ablation_reg_${lambda1}_${lambda2}"
    done
done
```

### Ablation 3: Time Scheduling

```bash
for schedule in linear cosine sigmoid exponential; do
    python scripts/train.py \
        --config configs/quadratic.yaml \
        --model.time_schedule $schedule \
        --training.run_name "ablation_schedule_${schedule}"
done
```

### Ablation 4: Pullback Metric Approximation

Compare exact vs. approximate pullback metrics:

```python
# In src/models/vae.py, modify compute_pullback_metric:
# approximate=False for exact computation
# approximate=True for Monte Carlo approximation

# Measure impact on path length regularization
```

---

## Analysis and Visualization

### Trajectory Visualization

Visualize trajectories in latent space:

```bash
python scripts/visualize_trajectories.py \
    --checkpoint checkpoints/nrf_spline_best.pt \
    --prompts "A red cube on top of a blue sphere" \
    --num_points 20 \
    --output figures/trajectory_visualization.png
```

### Failure Case Analysis

Generate images for challenging prompts:

```bash
python scripts/sample.py \
    --checkpoint checkpoints/nrf_spline_best.pt \
    --prompts_file data/challenging_prompts.txt \
    --steps 4 \
    --num_samples 16 \
    --output figures/failure_cases/
```

### Path Length Analysis

Compute path lengths for different teachers:

```bash
python scripts/analyze_paths.py \
    --checkpoints checkpoints/nrf_*.pt \
    --num_samples 1000 \
    --output results/path_analysis.json
```

---

## Computational Requirements

### Training

| Method | GPUs | Batch Size | Time | GPU-Hours |
|--------|------|------------|------|-----------|
| Linear RF | 4× A100 | 64 | 36h | 144 |
| Quadratic | 4× A100 | 64 | 38h | 152 |
| Cubic Spline | 4× A100 | 48 | 48h | 192 |
| Schrödinger Bridge | 8× A100 | 32 | 72h | 576 |

### Evaluation

| Task | GPUs | Time |
|------|------|------|
| FID/IS (5K samples) | 1× A100 | 1h |
| CLIPScore (5K samples) | 1× A100 | 0.5h |
| Compositional (350 prompts) | 1× A100 | 0.5h |

---

## Reproducing Paper Results

### Table 1: Standard Metrics

```bash
# Generate all results for Table 1
bash scripts/reproduce_table1.sh
```

This script runs:
1. DDIM baseline (50 steps)
2. DPM-Solver++ baseline (20 steps)
3. Linear RF (4, 8 steps)
4. All NRF variants (4 steps)

### Table 2: Compositional Evaluation

```bash
bash scripts/reproduce_table2.sh
```

### Table 3: Step Budget Ablation

```bash
bash scripts/reproduce_table3.sh
```

### Figure 2: Trajectory Visualization

```bash
python scripts/generate_figure2.py
```

### Figure 3: Qualitative Comparison

```bash
python scripts/generate_figure3.py
```

---

## Tips and Troubleshooting

### Memory Issues

If you encounter OOM errors:

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 32  # or 16
   ```

2. **Use gradient accumulation**:
   ```yaml
   training:
     batch_size: 16
     gradient_accumulation_steps: 4  # effective batch size = 64
   ```

3. **Disable pullback metric computation**:
   ```yaml
   teacher:
     path_length_weight: 0.0
     curvature_weight: 0.0
   ```

### Slow Training

1. **Use mixed precision** (enabled by default):
   ```yaml
   training:
     use_amp: true
   ```

2. **Precompute text embeddings**:
   ```bash
   python -c "from src.data.datasets import precompute_text_embeddings; ..."
   ```

3. **Use WebDataset for large datasets**:
   ```yaml
   data:
     dataset: laion
     use_webdataset: true
   ```

### Unstable Training

1. **Reduce learning rate**:
   ```yaml
   training:
     learning_rate: 5.0e-5
   ```

2. **Enable gradient clipping**:
   ```yaml
   training:
     grad_clip: 1.0
   ```

3. **Increase EMA decay**:
   ```yaml
   training:
     ema_decay: 0.9999
   ```

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{zhou2025nonlinear,
  title={Nonlinear Rectified Flows for AI Image Generation},
  author={Zhou, Todd Y.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

---

## Contact

For questions or issues, please:
1. Open an issue on GitHub
2. Email: your-email@example.com
3. Join our Discord: [link]

---

## Acknowledgments

This research was supported by the Kempner Undergraduate Research Experience (KURE) at Harvard University. We thank the Kempner Institute for providing computational resources and fostering a vibrant research community.
