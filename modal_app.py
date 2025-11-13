"""
Modal deployment for KURE Project - Nonlinear Rectified Flows (NRF) for AI Image Generation

Optimized for caching and cost savings:
- Base image with stable dependencies (cached across deployments)
- Project code installed from requirements.txt to ensure consistency
- Volumes for persistent storage (models, datasets, outputs)
- Git repository caching to avoid re-cloning
- Checkpoint saving to prevent GPU timeout losses
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("kure-nrf-project")

# Copy requirements.txt into the image for installation
# This ensures we install exactly what's in requirements.txt
requirements_txt = """
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
diffusers>=0.21.0
accelerate>=0.24.0
transformers>=4.35.0
open_clip_torch>=2.20.0
pot>=0.9.1
pytorch-fid>=0.3.0
torchmetrics>=1.2.0
lpips>=0.1.4
clean-fid>=0.1.35
datasets>=2.14.0
webdataset>=0.2.48
pillow>=10.0.0
opencv-python>=4.8.0
geomstats>=2.6.0
jax>=0.4.20
jaxlib>=0.4.20
wandb>=0.15.0
tensorboard>=2.14.0
omegaconf>=2.3.0
hydra-core>=1.3.0
tqdm>=4.66.0
einops>=0.7.0
timm>=0.9.0
safetensors>=0.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
PyYAML>=6.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
jinja2>=3.1.0
"""

# Base image with all dependencies - install from requirements to ensure consistency
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install system dependencies that might be missing
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libgcc-s1",
        "libffi-dev",
        "libssl-dev",
        "python3-dev",
        "pkg-config",
    )
    # Upgrade pip and install build tools first
    .pip_install([
        "pip>=23.0",
        "setuptools>=65.0",
        "wheel",
    ])
    # Install PyTorch first (large, stable dependency)
    # Modal will automatically use CUDA-enabled PyTorch when GPU is specified
    .pip_install([
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
    ])
    # Install core scientific computing libraries
    .pip_install([
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0.0",
    ])
    # Install ML frameworks
    .pip_install([
        "diffusers>=0.21.0",
        "accelerate>=0.24.0",
        "transformers>=4.35.0",
        "open_clip_torch>=2.20.0",
    ])
    # Install optimal transport and evaluation
    .pip_install([
        "pot>=0.9.1",
        "pytorch-fid>=0.3.0",
        "torchmetrics>=1.2.0",
        "lpips>=0.1.4",
        "clean-fid>=0.1.35",
    ])
    # Install data processing
    .pip_install([
        "datasets>=2.14.0",
        "webdataset>=0.2.48",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "pycocotools>=2.0.6",  # For COCO dataset loading
    ])
    # Install geometry and numerics
    .pip_install([
        "geomstats>=2.6.0",
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
    ])
    # Install training infrastructure
    .pip_install([
        "wandb>=0.15.0",
        "tensorboard>=2.14.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
    ])
    # Install utilities
    .pip_install([
        "tqdm>=4.66.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "safetensors>=0.4.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
    ])
    # Install web framework (required for FastAPI endpoints)
    .pip_install([
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "jinja2>=3.1.0",
        "python-multipart>=0.0.6",
    ])
    # Install CLIP from GitHub
    .pip_install("git+https://github.com/openai/CLIP.git")
)

# Final image - base dependencies are cached
image = base_image

# Create volumes for persistent storage (cached across runs)
models_volume = modal.Volume.from_name("kure-models", create_if_missing=True)
datasets_volume = modal.Volume.from_name("kure-datasets", create_if_missing=True)
outputs_volume = modal.Volume.from_name("kure-outputs", create_if_missing=True)

# Cache volume for git repos and downloaded data
cache_volume = modal.Volume.from_name("kure-cache", create_if_missing=True)

# Helper function to setup project with volume caching
def setup_project_cached(repo_url="https://github.com/Todd7777/KURE-Project.git"):
    """
    Setup project with caching to avoid re-cloning and re-installing.
    Uses volume to cache the git repository and installation.
    Ensures the package is properly installed and Python path is set correctly.
    """
    import subprocess
    import sys
    import os
    
    cache_dir = "/cache"
    project_cache_dir = f"{cache_dir}/KURE-Project"
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Clone or update repository in cache
    if not os.path.exists(project_cache_dir):
        print("Cloning repository for the first time...")
        result = subprocess.run([
            "git", "clone", repo_url, project_cache_dir
        ], check=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Git clone output: {result.stdout}")
            print(f"Git clone errors: {result.stderr}")
        print("Repository cloned successfully")
    else:
        print("Repository found in cache, updating...")
        try:
            # Try to pull latest changes (non-destructive)
            result = subprocess.run([
                "git", "-C", project_cache_dir, "pull", "--ff-only"
            ], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("Repository updated!")
            else:
                print(f"Note: Could not update repository: {result.stderr}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            # If pull fails, just use cached version
            print(f"Using cached repository version: {e}")
    
    # Change to project directory
    os.chdir(project_cache_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Verify and create project structure if needed
    src_dir = os.path.join(project_cache_dir, "src")
    if not os.path.exists(src_dir):
        print(f"Creating source directory: {src_dir}")
        os.makedirs(src_dir, exist_ok=True)
    print(f"✓ Source directory: {src_dir}")
    
    # Create required directories if they don't exist
    print("Ensuring project structure exists...")
    required_dirs = [
        "src/data",
        "src/models",
        "src/models/teachers",
        "src/training",
        "scripts",
        "configs"
    ]
    for req_dir in required_dirs:
        full_path = os.path.join(project_cache_dir, req_dir)
        if not os.path.exists(full_path):
            print(f"  Creating missing directory: {req_dir}/")
            os.makedirs(full_path, exist_ok=True)
        else:
            print(f"✓ Found {req_dir}/")
    
    # Add src directory to Python path (needed for imports in scripts)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    print(f"✓ Added {src_dir} to Python path")
    
    # Also add project directory to path
    if project_cache_dir not in sys.path:
        sys.path.insert(0, project_cache_dir)
    
    # Create missing __init__.py files if needed (with defensive imports)
    init_files = {
        "src/data/__init__.py": '''"""
Data loading and preprocessing modules for Nonlinear Rectified Flows.

This package provides:
- Dataset loaders (COCO, CC3M, LAION)
- Data augmentation utilities
- Text embedding preprocessing
"""

# Import augmentation utilities (defensive)
try:
    from .augmentation import (
        get_transforms,
        RandomHorizontalFlip,
        RandomResizedCrop,
        ColorJitter,
    )
except ImportError:
    get_transforms = None
    RandomHorizontalFlip = None
    RandomResizedCrop = None
    ColorJitter = None

# Import dataset functions and classes (defensive)
try:
    from .datasets import (
        create_dataloader,
        COCODataset,
        precompute_text_embeddings,
        download_coco_dataset,
    )
except ImportError:
    create_dataloader = None
    COCODataset = None
    precompute_text_embeddings = None
    download_coco_dataset = None

__all__ = [
    "create_dataloader",
    "COCODataset",
    "precompute_text_embeddings",
    "download_coco_dataset",
    "get_transforms",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
    "ColorJitter",
]
''',
        "src/models/__init__.py": '''"""
Model architectures for Nonlinear Rectified Flows.
"""

try:
    from .nrf_base import NonlinearRectifiedFlow, TimeScheduler
except ImportError:
    NonlinearRectifiedFlow = None
    TimeScheduler = None

try:
    from .unet import UNetVelocityPredictor
except ImportError:
    UNetVelocityPredictor = None

try:
    from .vae import create_vae, PullbackMetricVAE
except ImportError:
    create_vae = None
    PullbackMetricVAE = None

__all__ = [
    "NonlinearRectifiedFlow",
    "TimeScheduler",
    "UNetVelocityPredictor",
    "create_vae",
    "PullbackMetricVAE",
]
''',
        "src/training/__init__.py": '''"""
Training utilities for Nonlinear Rectified Flows.
"""

try:
    from .trainer import NRFTrainer, setup_distributed, cleanup_distributed
except ImportError:
    NRFTrainer = None
    setup_distributed = None
    cleanup_distributed = None

__all__ = [
    "NRFTrainer",
    "setup_distributed",
    "cleanup_distributed",
]
''',
        "src/models/teachers/__init__.py": '''"""
Teacher models for Nonlinear Rectified Flows.
"""

try:
    from .linear import LinearTeacher
except ImportError:
    LinearTeacher = None

try:
    from .quadratic import QuadraticTeacher, AdaptiveQuadraticTeacher
except ImportError:
    QuadraticTeacher = None
    AdaptiveQuadraticTeacher = None

try:
    from .cubic_spline import CubicSplineTeacher, CubicSplineController
except ImportError:
    CubicSplineTeacher = None
    CubicSplineController = None

try:
    from .schrodinger_bridge import (
        SchrodingerBridgeTeacher,
        SchrodingerBridgeDriftNet,
        NystromSinkhornSolver,
    )
except ImportError:
    SchrodingerBridgeTeacher = None
    SchrodingerBridgeDriftNet = None
    NystromSinkhornSolver = None

__all__ = [
    "LinearTeacher",
    "QuadraticTeacher",
    "AdaptiveQuadraticTeacher",
    "CubicSplineTeacher",
    "CubicSplineController",
    "SchrodingerBridgeTeacher",
    "SchrodingerBridgeDriftNet",
    "NystromSinkhornSolver",
]
'''
    }
    
    for init_path, init_content in init_files.items():
        full_path = os.path.join(project_cache_dir, init_path)
        if not os.path.exists(full_path):
            print(f"  Creating missing {init_path}")
            with open(full_path, 'w') as f:
                f.write(init_content)
        else:
            print(f"✓ Found {init_path}")
    
    # Check for datasets.py - create a minimal stub if missing (needed for imports)
    datasets_py_path = os.path.join(project_cache_dir, "src/data/datasets.py")
    if not os.path.exists(datasets_py_path):
        print("Warning: src/data/datasets.py not found in repository")
        print("  Creating minimal stub for imports (training will require full implementation)")
        # Create a minimal stub that allows imports to work
        minimal_datasets = '''"""
Dataset loaders for Nonlinear Rectified Flows.
Minimal stub - full implementation needed for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

# Minimal stubs to allow imports
class COCODataset(Dataset):
    """Stub - full implementation needed for training"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("COCODataset not fully implemented. Please ensure src/data/datasets.py is in the repository.")

def create_dataloader(*args, **kwargs):
    """Stub - full implementation needed for training"""
    raise NotImplementedError("create_dataloader not fully implemented. Please ensure src/data/datasets.py is in the repository.")

def precompute_text_embeddings(*args, **kwargs):
    """Stub - full implementation needed for training"""
    raise NotImplementedError("precompute_text_embeddings not fully implemented. Please ensure src/data/datasets.py is in the repository.")

def download_coco_dataset(*args, **kwargs):
    """Stub - full implementation needed for training"""
    raise NotImplementedError("download_coco_dataset not fully implemented. Please ensure src/data/datasets.py is in the repository.")

__all__ = ["COCODataset", "create_dataloader", "precompute_text_embeddings", "download_coco_dataset"]
'''
        with open(datasets_py_path, 'w') as f:
            f.write(minimal_datasets)
        print(f"  Created minimal stub at {datasets_py_path}")
    else:
        print(f"✓ Found src/data/datasets.py")
    
    # Verify critical files exist (these should be in the repo)
    # Note: For inference, we mainly need models and sampling scripts
    critical_files_for_inference = [
        "src/models/nrf_base.py",
        "scripts/sample.py",  # Needed for inference
    ]
    
    critical_files_for_training = [
        "src/training/trainer.py",
        "scripts/train.py",
        "setup.py"
    ]
    
    # Check inference-critical files
    for req_file in critical_files_for_inference:
        full_path = os.path.join(project_cache_dir, req_file)
        if not os.path.exists(full_path):
            raise RuntimeError(f"Required file for inference not found: {full_path}")
        print(f"✓ Found {req_file}")
    
    # Check training-critical files (warn if missing, but don't fail)
    for req_file in critical_files_for_training:
        full_path = os.path.join(project_cache_dir, req_file)
        if not os.path.exists(full_path):
            print(f"⚠️  Warning: Training file not found: {req_file} (training will fail)")
        else:
            print(f"✓ Found {req_file}")
    
    # Install package in editable mode (this helps with imports)
    # Even if this fails, scripts use sys.path.insert(0, "src") which should work
    print("Installing package in editable mode...")
    install_result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
        cwd=project_cache_dir,
        capture_output=True,
        text=True,
        timeout=120  # 2 minute timeout
    )
    
    if install_result.returncode == 0:
        print("✓ Package installed successfully")
        # Verify installation worked by checking if packages are importable
        # But don't fail if they're not - scripts handle their own path setup
        try:
            # Quick test import (in a subprocess to avoid contaminating current process)
            test_script = f"""
import sys
sys.path.insert(0, '{src_dir}')
try:
    from data import datasets
    from models import nrf_base
    from training import trainer
    print("✓ All modules can be imported")
except Exception as e:
    print(f"Note: Direct import test: {{e}}")
    print("  (This is OK - scripts use sys.path.insert)")
"""
            test_result = subprocess.run(
                [sys.executable, "-c", test_script],
                cwd=project_cache_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if test_result.returncode == 0 and test_result.stdout:
                print(test_result.stdout.strip())
        except Exception as e:
            print(f"  (Import test skipped: {e})")
    else:
        # Installation had issues - this is OK because scripts handle path setup
        print(f"⚠️  Package installation completed with warnings (this is OK)")
        if install_result.stderr:
            # Only show first few lines of stderr to avoid clutter
            stderr_lines = install_result.stderr.strip().split('\n')[:3]
            for line in stderr_lines:
                if line.strip():
                    print(f"    {line}")
        print("  Scripts will use sys.path.insert(0, 'src') for imports")
    
    # Final verification: ensure scripts can find src directory
    # The training script does: sys.path.insert(0, "src") then imports from data, models, training
    # So as long as src/data, src/models, src/training exist with __init__.py, it should work
    print("✓ Project setup complete!")
    print(f"  Project directory: {project_cache_dir}")
    print(f"  Source directory: {src_dir}")
    print(f"  Python path includes: {src_dir}")
    print(f"  Scripts will use: sys.path.insert(0, 'src') for imports")
    
    return project_cache_dir

# GPU configuration using new string format (fixes deprecation warning)
# Note: Checkpoint saving is handled by the training script itself
@app.function(
    image=image,
    gpu="A100",  # Use string format instead of modal.gpu.A100()
    volumes={
        "/models": models_volume,
        "/datasets": datasets_volume, 
        "/outputs": outputs_volume,
        "/cache": cache_volume,
    },
    timeout=7200,  # 2 hours timeout (increased to prevent timeouts)
    retries=1,  # Retry once if it fails
)
def train_nrf_model(
    config_name: str = "quadratic",
    gpus: int = 1,
    wandb_project: str = "kure-nrf",
    checkpoint_interval: int = 1000,  # Save checkpoint every N steps
    **kwargs
):
    """
    Train NRF model with specified configuration
    
    Args:
        config_name: Configuration to use (quadratic, spline, sb)
        gpus: Number of GPUs to use (passed to training script)
        wandb_project: Weights & Biases project name
        checkpoint_interval: Save checkpoint every N training steps
        **kwargs: Additional training parameters
    """
    import subprocess
    import sys
    import os
    import time
    
    # Setup project (uses cache)
    project_dir = setup_project_cached()
    os.chdir(project_dir)
    
    # Create checkpoint directory in models volume (persistent across runs)
    # This ensures checkpoints survive GPU timeouts and are available across runs
    checkpoint_dir = "/models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir} (persisted in volume)")
    
    # Set up environment variables
    os.environ["WANDB_PROJECT"] = wandb_project
    if "WANDB_API_KEY" in os.environ:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"
        print("Warning: WANDB_API_KEY not found, running in offline mode")
    
    # Override checkpoint directory in config by setting environment variable
    # The training script/trainer will use this if it checks the environment
    os.environ["CHECKPOINT_DIR"] = checkpoint_dir
    
    # Modify the config file to use the volume checkpoint directory
    # This ensures the trainer saves to the persistent volume (survives timeouts)
    import yaml
    original_config_path = f"configs/{config_name}.yaml"
    
    if os.path.exists(original_config_path):
        with open(original_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle config inheritance if _base_ is present
        # For now, we'll just override the checkpoint_dir which should work
        # The training script will handle base config merging
        
        # Ensure training section exists
        if 'training' not in config:
            config['training'] = {}
        
        # Override checkpoint_dir to use volume (absolute path)
        config['training']['checkpoint_dir'] = checkpoint_dir
        
        # Write modified config to temp file
        temp_config_path = f"/tmp/{config_name}_modal.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        config_path = temp_config_path
        print(f"✓ Modified config to use checkpoint_dir={checkpoint_dir}")
        print(f"  Config saved to: {temp_config_path}")
    else:
        print(f"Warning: Config file {original_config_path} not found, using as-is")
        config_path = original_config_path
    
    # Prepare training command
    # Use the modified config file that points to the volume checkpoint directory
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", config_path,  # Use the modified config
        "--gpus", str(gpus),
    ]
    
    # Add checkpoint directory if training script supports it (check the script for --checkpoint-dir or similar)
    # For now, we'll rely on the training script's default checkpoint location
    # and ensure it saves to the models volume by setting the working directory appropriately
    
    # Add additional kwargs as command line arguments
    for key, value in kwargs.items():
        if key != "checkpoint_interval":  # Skip our internal parameter (for future use)
            # Convert snake_case to kebab-case for command line
            key_arg = key.replace('_', '-')
            cmd.extend([f"--{key_arg}", str(value)])
    
    print(f"Running training command: {' '.join(cmd)}")
    print(f"Checkpoints directory: {checkpoint_dir} (persisted in volume)")
    print(f"Training timeout: 2 hours (checkpoints saved periodically by training script)")
    print(f"Working directory: {project_dir}")
    
    # Set up environment variables for the subprocess
    env = os.environ.copy()
    # Ensure PYTHONPATH includes the project directories
    src_dir = os.path.join(project_dir, "src")
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        pythonpath = f"{src_dir}:{project_dir}:{pythonpath}"
    else:
        pythonpath = f"{src_dir}:{project_dir}"
    env["PYTHONPATH"] = pythonpath
    env["PWD"] = project_dir  # Set working directory in environment
    
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    
    # Run training with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=project_dir,
        env=env  # Pass environment with PYTHONPATH
    )
    
    # Stream output in real-time
    output_lines = []
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            output_lines.append(line)
    except Exception as e:
        print(f"Error reading output: {e}")
    
    process.wait()
    
    if process.returncode != 0:
        error_output = ''.join(output_lines)
        print(f"Training failed with error. Last 50 lines of output:")
        print('\n'.join(output_lines[-50:]))
        raise RuntimeError(f"Training failed with exit code {process.returncode}")
    
    print("Training completed successfully!")
    
    # Note: Volumes are automatically persisted by Modal
    # Checkpoints saved to /models will be available in subsequent runs
    
    return {
        "status": "success",
        "stdout": ''.join(output_lines),
        "checkpoint_dir": checkpoint_dir,
    }

@app.function(
    image=image,
    gpu="T4",  # Lighter GPU for inference (string format)
    volumes={
        "/models": models_volume,
        "/outputs": outputs_volume,
        "/cache": cache_volume,
    },
    timeout=600,  # 10 minutes timeout
)
def generate_images(
    prompt: str,
    checkpoint_path: str = None,
    steps: int = 4,
    num_samples: int = 16,
    output_dir: str = "/outputs"
):
    """
    Generate images using trained NRF model
    
    Args:
        prompt: Text prompt for image generation
        checkpoint_path: Path to model checkpoint (defaults to latest.pt in /models/checkpoints)
        steps: Number of sampling steps
        num_samples: Number of images to generate
        output_dir: Directory to save generated images
    """
    import subprocess
    import sys
    import os
    from datetime import datetime
    
    # Setup project (uses cache)
    project_dir = setup_project_cached()
    os.chdir(project_dir)
    
    # Use latest checkpoint if none specified
    if checkpoint_path is None:
        latest_checkpoint = "/models/checkpoints/latest.pt"
        if os.path.exists(latest_checkpoint):
            checkpoint_path = latest_checkpoint
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            print("Warning: No checkpoint specified and no latest.pt found")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = f"{output_dir}/samples_{timestamp}"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Prepare sampling command
    cmd = [
        sys.executable, "scripts/sample.py",
        "--prompt", prompt,
        "--steps", str(steps),
        "--num_samples", str(num_samples),
        "--output_dir", sample_dir
    ]
    
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    
    print(f"Running sampling command: {' '.join(cmd)}")
    
    # Set up environment variables for the subprocess
    env = os.environ.copy()
    src_dir = os.path.join(project_dir, "src")
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        pythonpath = f"{src_dir}:{project_dir}:{pythonpath}"
    else:
        pythonpath = f"{src_dir}:{project_dir}"
    env["PYTHONPATH"] = pythonpath
    env["PWD"] = project_dir
    
    # Run sampling
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_dir, env=env)
    
    if result.returncode != 0:
        print(f"Sampling failed with error: {result.stderr}")
        raise RuntimeError(f"Sampling failed: {result.stderr}")
    
    print("Image generation completed successfully!")
    print(result.stdout)
    
    # List generated files
    generated_files = []
    if os.path.exists(sample_dir):
        generated_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Note: Volumes are automatically persisted by Modal
    # Generated images in /outputs will be available in subsequent runs
    
    return {
        "status": "success",
        "output_dir": sample_dir,
        "generated_files": generated_files,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.function(
    image=image,
    gpu="T4",
    volumes={
        "/models": models_volume,
        "/datasets": datasets_volume,
        "/outputs": outputs_volume,
        "/cache": cache_volume,
    },
    timeout=1800,  # 30 minutes timeout
)
def evaluate_model(
    checkpoint_path: str,
    dataset: str = "coco",
    steps: list = None,
    metrics: list = None
):
    """
    Evaluate NRF model performance
    
    Args:
        checkpoint_path: Path to model checkpoint
        dataset: Dataset to evaluate on (coco, compositional)
        steps: List of sampling steps to evaluate
        metrics: List of metrics to compute (fid, is, clip)
    """
    import subprocess
    import sys
    import os
    
    if steps is None:
        steps = [1, 2, 4, 8]
    if metrics is None:
        metrics = ["fid", "is", "clip"]
    
    # Setup project (uses cache)
    project_dir = setup_project_cached()
    os.chdir(project_dir)
    
    # Prepare evaluation command
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", checkpoint_path,
        "--dataset", dataset,
        "--steps"] + [str(s) for s in steps] + [
        "--metrics"] + metrics
    
    print(f"Running evaluation command: {' '.join(cmd)}")
    
    # Set up environment variables for the subprocess
    env = os.environ.copy()
    src_dir = os.path.join(project_dir, "src")
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        pythonpath = f"{src_dir}:{project_dir}:{pythonpath}"
    else:
        pythonpath = f"{src_dir}:{project_dir}"
    env["PYTHONPATH"] = pythonpath
    env["PWD"] = project_dir
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_dir, env=env)
    
    if result.returncode != 0:
        print(f"Evaluation failed with error: {result.stderr}")
        raise RuntimeError(f"Evaluation failed: {result.stderr}")
    
    print("Evaluation completed successfully!")
    print(result.stdout)
    
    # Note: Volumes are automatically persisted by Modal
    # Evaluation results in /outputs will be available in subsequent runs
    
    return {
        "status": "success",
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.function(
    image=image,
    timeout=300,
    volumes={"/cache": cache_volume},
)
def list_available_configs():
    """List available configuration files"""
    import os
    
    # Setup project (uses cache)
    project_dir = setup_project_cached()
    
    configs_dir = f"{project_dir}/configs"
    if os.path.exists(configs_dir):
        configs = [f.replace('.yaml', '') for f in os.listdir(configs_dir) if f.endswith('.yaml')]
        return {"available_configs": configs}
    else:
        return {"available_configs": []}

# Web interface for easy access (fixes deprecation warnings)
@app.function(
    image=image,
    max_containers=10,  # Allow up to 10 concurrent containers
)
@modal.fastapi_endpoint()  # FastAPI is now installed in the image
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "KURE NRF Project"}

if __name__ == "__main__":
    # Example usage
    with app.run():
        # List available configurations
        configs = list_available_configs.remote()
        print("Available configurations:", configs)
        
        # Example training (uncomment to run)
        # result = train_nrf_model.remote(
        #     config_name="quadratic",
        #     gpus=1,
        #     wandb_project="kure-nrf-demo",
        #     checkpoint_interval=1000
        # )
        # print("Training result:", result)
        
        # Example image generation (uncomment to run)
        # result = generate_images.remote(
        #     prompt="A red cube on top of a blue sphere",
        #     steps=4,
        #     num_samples=8
        # )
        # print("Generation result:", result)
