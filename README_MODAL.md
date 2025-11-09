# KURE Project - Modal Deployment

Welcome to the Modal deployment of the KURE Project! This setup allows anyone to run the Nonlinear Rectified Flows (NRF) AI image generation system on Todd's Modal account.

## 🚀 Quick Start (For Users)

### 1. Prerequisites
- Python 3.10+
- Modal CLI access (will be provided)

### 2. One-Command Setup
```bash
python deploy_to_modal.py
```

This script will:
- ✅ Check Modal installation and authentication
- 🔐 Set up API secrets (optional)
- 🚀 Deploy the main Modal app
- 🌐 Deploy the web interface
- 🧪 Test the deployment

### 3. Start Using the System

**Generate Images:**
```bash
python run_inference.py --prompt "A futuristic city at sunset"
```

**Train Models:**
```bash
python run_training.py --config quadratic --gpus 1
```

**Evaluate Models:**
```bash
python run_evaluation.py --checkpoint /models/my_model.pt --dataset coco
```

## 📁 File Structure

```
KURE-Project/
├── 🚀 Modal Deployment Files
│   ├── modal_app.py              # Main Modal application
│   ├── modal_web_interface.py    # Web UI for image generation
│   ├── modal_requirements.txt    # Modal-specific dependencies
│   └── deploy_to_modal.py        # One-click deployment script
│
├── 🎮 User Scripts
│   ├── run_training.py           # Easy training interface
│   ├── run_inference.py          # Easy image generation
│   ├── run_evaluation.py         # Easy model evaluation
│   └── setup_modal_secrets.py    # API keys setup
│
├── 📚 Documentation
│   ├── MODAL_DEPLOYMENT.md       # Detailed deployment guide
│   └── README_MODAL.md           # This file
│
└── 🔬 Original Project Files
    ├── src/                      # Source code
    ├── scripts/                  # Original scripts
    ├── configs/                  # Model configurations
    └── requirements.txt          # Original dependencies
```

## 🎯 Available Configurations

- **`quadratic`**: Quadratic teacher (fastest training)
- **`spline`**: Cubic spline with controller (balanced)
- **`sb`**: Schrödinger Bridge (highest quality)

## 💡 Example Workflows

### Quick Image Generation
```bash
# Generate 8 images with 4 sampling steps
python run_inference.py --prompt "A robot painting a masterpiece" --steps 4 --num-samples 8
```

### Full Training Pipeline
```bash
# 1. Train a model
python run_training.py --config spline --gpus 4 --epochs 100

# 2. Generate test images
python run_inference.py --prompt "Test generation" --steps 4

# 3. Evaluate performance
python run_evaluation.py --checkpoint /models/latest.pt --dataset coco
```

### Batch Processing
```bash
# Generate multiple image sets
python run_inference.py --prompt "Sunset landscape" --num-samples 16
python run_inference.py --prompt "Abstract art" --num-samples 16
python run_inference.py --prompt "Futuristic architecture" --num-samples 16
```

## 🔧 Advanced Usage

### Direct Modal Function Calls
```python
import modal
from modal_app import app, generate_images

with app.run():
    result = generate_images.remote(
        prompt="A beautiful landscape",
        steps=8,
        num_samples=16
    )
    print(result)
```

### Custom Training Parameters
```bash
python run_training.py \
  --config quadratic \
  --epochs 200 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --wandb-project my-experiment
```

## 🌐 Web Interface

After deployment, you can access a user-friendly web interface directly from the Modal dashboard. The web interface allows you to:

- 🎨 Generate images with custom prompts
- ⚙️ Adjust sampling parameters
- 📸 View and download results
- 🚀 No coding required!

## 💰 Cost Information

**Estimated costs on Modal:**
- **Training**: ~$1-5/hour (depending on GPU type)
- **Image Generation**: ~$0.10-0.50 per 100 images
- **Evaluation**: ~$0.50-2.00 per run

**Cost optimization tips:**
- Use T4 GPUs for inference (cheaper)
- Use A100 GPUs only for training
- Batch multiple operations together
- Monitor usage in Modal dashboard

## 🛠️ Troubleshooting

### Common Issues

**"Modal not authenticated"**
```bash
modal token new
```

**"modal_app.py not found"**
- Make sure you're in the KURE-Project directory
- Run `ls` to verify files are present

**"Import errors"**
- The Modal environment handles all dependencies automatically
- No local installation required

**"GPU unavailable"**
- Try different GPU types in modal_app.py
- Check Modal dashboard for availability

### Getting Help

1. 📖 Read the detailed guide: `MODAL_DEPLOYMENT.md`
2. 🌐 Check Modal documentation: https://modal.com/docs
3. 💬 Contact project maintainer
4. 🐛 Report issues on GitHub

## 🎨 Example Prompts

Try these prompts for impressive results:

**Artistic:**
- "A surreal landscape with floating islands and waterfalls"
- "An abstract painting in the style of Kandinsky"
- "A minimalist geometric composition with vibrant colors"

**Realistic:**
- "A cozy coffee shop on a rainy day"
- "A modern architectural building with glass and steel"
- "A serene mountain lake at golden hour"

**Creative:**
- "A steampunk robot reading a book in a library"
- "A neon-lit cyberpunk street scene"
- "A magical forest with glowing mushrooms"

## 📊 Performance Metrics

The system supports comprehensive evaluation:

**Metrics:**
- **FID**: Fréchet Inception Distance
- **IS**: Inception Score  
- **CLIP**: CLIP Score for text-image alignment

**Datasets:**
- **COCO**: Standard benchmark dataset
- **Compositional**: Advanced compositional understanding

## 🔄 Updates and Maintenance

The Modal deployment automatically:
- 📦 Pulls the latest code from GitHub
- 🔧 Installs all required dependencies
- 💾 Manages persistent storage for models and outputs
- 🔄 Handles scaling and resource management

## 🎉 Success Stories

Once deployed, you can:
- ✨ Generate high-quality images in seconds
- 🚀 Train state-of-the-art models without local GPU requirements
- 📈 Scale to multiple GPUs for faster training
- 🌍 Share access with team members easily
- 💾 Automatically save and manage all outputs

---

**Ready to create amazing AI art? Start with:**
```bash
python deploy_to_modal.py
```

**Happy generating! 🎨✨**
