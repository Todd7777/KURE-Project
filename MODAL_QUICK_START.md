# 🚀 KURE Project - Modal Quick Start

**Get started with AI image generation in under 5 minutes!**

## One-Command Setup

```bash
# 1. Clone the repository
git clone https://github.com/Todd7777/KURE-Project.git
cd KURE-Project

# 2. Deploy to Modal (one command does everything!)
python deploy_to_modal.py

# 3. Start generating images immediately
python run_inference.py --prompt "A beautiful sunset over mountains"
```

That's it! 🎉

## What You Get

### 🎨 **Instant Image Generation**
```bash
# Quick generation (4 steps, 8 images)
python run_inference.py --prompt "A robot painting a masterpiece"

# High quality (8 steps, 16 images) 
python run_inference.py --prompt "Abstract art" --steps 8 --num-samples 16

# Batch processing
python run_inference.py --prompt "Futuristic architecture" --num-samples 32
```

### 🚂 **Model Training on Cloud GPUs**
```bash
# Fast training (quadratic teacher)
python run_training.py --config quadratic --gpus 1

# Best quality (cubic spline teacher)
python run_training.py --config spline --gpus 4

# Research mode (Schrödinger Bridge)
python run_training.py --config sb --gpus 8
```

### 📊 **Model Evaluation**
```bash
# Standard benchmarks
python run_evaluation.py --checkpoint /models/my_model.pt --dataset coco

# Compositional understanding
python run_evaluation.py --checkpoint /models/my_model.pt --dataset compositional
```

### 🌐 **Web Interface**
- Access from Modal dashboard after deployment
- No coding required - just type prompts and generate!
- Perfect for non-technical users

## 💰 Cost Estimates

- **Image Generation**: ~$0.10-0.50 per 100 images
- **Model Training**: ~$1-5 per hour
- **Evaluation**: ~$0.50-2.00 per run

*Pay only for what you use - no upfront costs!*

## 🎯 Example Prompts

Try these for amazing results:

**Artistic:**
- "A surreal landscape with floating islands"
- "An abstract painting in vibrant colors"
- "A minimalist geometric composition"

**Realistic:**
- "A cozy coffee shop on a rainy day"
- "Modern architecture with glass and steel"
- "A serene mountain lake at golden hour"

**Creative:**
- "A steampunk robot in a library"
- "A neon-lit cyberpunk street"
- "A magical forest with glowing mushrooms"

## 🔧 Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com) (free tier available)
2. **Modal CLI**: 
   ```bash
   pip install modal
   modal token new
   ```

That's all you need! No GPU, no complex setup, no local dependencies.

## 📚 Need More Help?

- **Detailed Guide**: [MODAL_DEPLOYMENT.md](MODAL_DEPLOYMENT.md)
- **Full Documentation**: [README_MODAL.md](README_MODAL.md)
- **GitHub Repository**: [Todd7777/KURE-Project](https://github.com/Todd7777/KURE-Project)

## 🎉 What Makes This Special?

- **Zero Setup**: No local GPU or complex installation
- **State-of-the-Art**: Nonlinear Rectified Flows for superior image quality
- **Multiple Teachers**: Quadratic, Cubic Spline, and Schrödinger Bridge
- **Fast Generation**: 1-16 sampling steps (as fast as 1 step!)
- **Web Interface**: User-friendly for everyone
- **Cost Effective**: Pay-per-use cloud computing
- **Persistent Storage**: Your models and outputs are saved
- **Auto Scaling**: Handles multiple users automatically

---

**Ready to create amazing AI art?**

```bash
git clone https://github.com/Todd7777/KURE-Project.git
cd KURE-Project
python deploy_to_modal.py
```

**Start generating in under 5 minutes! 🎨✨**
