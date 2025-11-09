# Modal Deployment Guide for KURE Project

This guide explains how to deploy and use the KURE Project (Nonlinear Rectified Flows for AI Image Generation) on Modal Labs.

## 🚀 Quick Start

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate
   ```bash
   pip install modal
   modal token new
   ```

### 1. Setup Secrets (Optional but Recommended)

Run the setup script to configure API keys:

```bash
python setup_modal_secrets.py
```

This will prompt you for:
- **Weights & Biases API Key** (for experiment tracking)
- **Hugging Face Token** (for accessing gated models)
- **OpenAI API Key** (for advanced evaluation metrics)

All secrets are optional - the system will work without them but with limited functionality.

### 2. Deploy the Modal App

```bash
modal deploy modal_app.py
```

This creates the Modal app with all necessary dependencies and GPU configurations.

## 📋 Available Functions

### Training

Train NRF models with different configurations:

```bash
# Basic training with quadratic teacher
python run_training.py --config quadratic

# Training with cubic spline teacher on multiple GPUs
python run_training.py --config spline --gpus 4

# Training with Schrödinger Bridge teacher
python run_training.py --config sb --gpus 8

# Custom training parameters
python run_training.py --config quadratic --epochs 100 --batch-size 32 --learning-rate 0.001
```

**Available configurations:**
- `quadratic`: Quadratic teacher configuration
- `spline`: Cubic spline with controller
- `sb`: Schrödinger Bridge (Entropic OT teacher)

### Image Generation

Generate images using trained models:

```bash
# Basic image generation
python run_inference.py --prompt "A red cube on top of a blue sphere"

# Advanced generation with custom parameters
python run_inference.py \
  --prompt "A futuristic city at sunset" \
  --steps 8 \
  --num-samples 32 \
  --checkpoint /models/nrf_spline_best.pt
```

**Parameters:**
- `--prompt`: Text description of the image to generate
- `--steps`: Number of sampling steps (1-16, default: 4)
- `--num-samples`: Number of images to generate (default: 16)
- `--checkpoint`: Path to trained model checkpoint (optional)

### Model Evaluation

Evaluate model performance on standard benchmarks:

```bash
# Evaluate on COCO captions
python run_evaluation.py \
  --checkpoint /models/nrf_spline_best.pt \
  --dataset coco \
  --steps 1 2 4 8 \
  --metrics fid is clip

# Evaluate on compositional suite
python run_evaluation.py \
  --checkpoint /models/nrf_spline_best.pt \
  --dataset compositional \
  --steps 4 \
  --metrics fid clip
```

**Available datasets:**
- `coco`: COCO captions dataset
- `compositional`: Compositional benchmark suite

**Available metrics:**
- `fid`: Fréchet Inception Distance
- `is`: Inception Score
- `clip`: CLIP Score

## 🔧 Advanced Usage

### Direct Modal Function Calls

You can also call Modal functions directly:

```python
import modal
from modal_app import app, train_nrf_model, generate_images, evaluate_model

with app.run():
    # Training
    result = train_nrf_model.remote(
        config_name="quadratic",
        gpus=1,
        wandb_project="my-nrf-experiment"
    )
    
    # Image generation
    result = generate_images.remote(
        prompt="A beautiful landscape",
        steps=4,
        num_samples=8
    )
    
    # Evaluation
    result = evaluate_model.remote(
        checkpoint_path="/models/my_model.pt",
        dataset="coco",
        steps=[1, 2, 4],
        metrics=["fid", "clip"]
    )
```

### Volume Management

The Modal app uses three persistent volumes:

- **`/models`**: Stores trained model checkpoints
- **`/datasets`**: Caches datasets for faster loading
- **`/outputs`**: Stores generated images and evaluation results

These volumes persist across function calls, so you don't need to re-download data.

### GPU Configuration

The app is configured with different GPU setups:

- **Training**: A100 GPUs (scalable to multiple GPUs)
- **Inference**: T4 GPUs (cost-effective for generation)
- **Evaluation**: T4 GPUs (sufficient for metrics computation)

You can modify the GPU configuration in `modal_app.py` if needed.

## 🎯 Example Workflows

### Complete Training Pipeline

```bash
# 1. Train a model
python run_training.py --config spline --gpus 4 --epochs 200

# 2. Generate sample images
python run_inference.py --prompt "A robot in a garden" --steps 4 --num-samples 16

# 3. Evaluate the model
python run_evaluation.py --checkpoint /models/latest_checkpoint.pt --dataset coco
```

### Batch Image Generation

```bash
# Generate multiple image sets with different prompts
python run_inference.py --prompt "A sunset over mountains" --num-samples 8
python run_inference.py --prompt "A futuristic car" --num-samples 8
python run_inference.py --prompt "An abstract painting" --num-samples 8
```

### Hyperparameter Sweep

```bash
# Train with different learning rates
python run_training.py --config quadratic --learning-rate 0.0001 --wandb-project nrf-sweep-1
python run_training.py --config quadratic --learning-rate 0.001 --wandb-project nrf-sweep-2
python run_training.py --config quadratic --learning-rate 0.01 --wandb-project nrf-sweep-3
```

## 🔍 Monitoring and Debugging

### Weights & Biases Integration

If you set up W&B secrets, all training runs will be automatically logged to your W&B project. You can monitor:

- Training loss curves
- Generated sample images
- Model checkpoints
- Hyperparameter configurations

### Modal Dashboard

Monitor your function calls, GPU usage, and costs at [modal.com/apps](https://modal.com/apps).

### Logs and Outputs

All function outputs are captured and returned. Check the console output for detailed logs and error messages.

## 💰 Cost Optimization

### Tips for Reducing Costs

1. **Use appropriate GPU types**: T4 for inference, A100 for training
2. **Batch operations**: Generate multiple images in a single call
3. **Use volumes**: Avoid re-downloading datasets
4. **Monitor usage**: Check the Modal dashboard regularly
5. **Set timeouts**: Functions have reasonable timeouts to prevent runaway costs

### Estimated Costs (as of 2024)

- **Training**: ~$1-5 per hour (depending on GPU type and count)
- **Inference**: ~$0.10-0.50 per 100 images
- **Evaluation**: ~$0.50-2.00 per evaluation run

## 🛠️ Troubleshooting

### Common Issues

1. **Import errors**: Make sure `modal_app.py` is in your current directory
2. **Authentication errors**: Run `modal token new` to re-authenticate
3. **GPU availability**: Try different GPU types if A100s are unavailable
4. **Memory errors**: Reduce batch size or number of samples

### Getting Help

1. Check the [Modal documentation](https://modal.com/docs)
2. Review the original KURE Project repository
3. Check Modal community forums
4. Contact the project maintainer

## 📚 Additional Resources

- [Modal Labs Documentation](https://modal.com/docs)
- [KURE Project Repository](https://github.com/Todd7777/KURE-Project)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)

---

**Happy generating! 🎨✨**
