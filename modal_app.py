"""
Modal deployment for KURE Project - Nonlinear Rectified Flows (NRF) for AI Image Generation
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("kure-nrf-project")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "curl",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libgcc-s1"
    )
    .pip_install([
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "diffusers>=0.21.0",
        "accelerate>=0.24.0",
        "transformers>=4.35.0",
        "open_clip_torch>=2.20.0",
        "pot>=0.9.1",
        "pytorch-fid>=0.3.0",
        "torchmetrics>=1.2.0",
        "lpips>=0.1.4",
        "clean-fid>=0.1.35",
        "datasets>=2.14.0",
        "webdataset>=0.2.48",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "geomstats>=2.6.0",
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "wandb>=0.15.0",
        "tensorboard>=2.14.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "safetensors>=0.4.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0"
    ])
    .pip_install("git+https://github.com/openai/CLIP.git")
    .run_commands([
        "pip install ot-pytorch",  # Install separately as it might have conflicts
    ])
)

# Create volumes for persistent storage
models_volume = modal.Volume.from_name("kure-models", create_if_missing=True)
datasets_volume = modal.Volume.from_name("kure-datasets", create_if_missing=True)
outputs_volume = modal.Volume.from_name("kure-outputs", create_if_missing=True)

# GPU configuration for training
GPU_CONFIG = modal.gpu.A100(count=1)  # Can be scaled up for multi-GPU training

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        "/models": models_volume,
        "/datasets": datasets_volume, 
        "/outputs": outputs_volume
    },
    timeout=3600,  # 1 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret", required=False)]
)
def train_nrf_model(
    config_name: str = "quadratic",
    gpus: int = 1,
    wandb_project: str = "kure-nrf",
    **kwargs
):
    """
    Train NRF model with specified configuration
    
    Args:
        config_name: Configuration to use (quadratic, spline, sb)
        gpus: Number of GPUs to use
        wandb_project: Weights & Biases project name
        **kwargs: Additional training parameters
    """
    import subprocess
    import sys
    import os
    
    # Clone the repository if not exists
    if not os.path.exists("/tmp/KURE-Project"):
        subprocess.run([
            "git", "clone", 
            "https://github.com/Todd7777/KURE-Project.git",
            "/tmp/KURE-Project"
        ], check=True)
    
    os.chdir("/tmp/KURE-Project")
    
    # Install the package
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Set up environment variables
    os.environ["WANDB_PROJECT"] = wandb_project
    if "WANDB_API_KEY" in os.environ:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"
        print("Warning: WANDB_API_KEY not found, running in offline mode")
    
    # Prepare training command
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", f"configs/{config_name}.yaml",
        "--gpus", str(gpus)
    ]
    
    # Add additional kwargs as command line arguments
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Running training command: {' '.join(cmd)}")
    
    # Run training
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Training failed with error: {result.stderr}")
        raise RuntimeError(f"Training failed: {result.stderr}")
    
    print("Training completed successfully!")
    print(result.stdout)
    
    return {
        "status": "success",
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.function(
    image=image,
    gpu=modal.gpu.T4(),  # Lighter GPU for inference
    volumes={
        "/models": models_volume,
        "/outputs": outputs_volume
    },
    timeout=600  # 10 minutes timeout
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
        checkpoint_path: Path to model checkpoint
        steps: Number of sampling steps
        num_samples: Number of images to generate
        output_dir: Directory to save generated images
    """
    import subprocess
    import sys
    import os
    from datetime import datetime
    
    # Clone the repository if not exists
    if not os.path.exists("/tmp/KURE-Project"):
        subprocess.run([
            "git", "clone", 
            "https://github.com/Todd7777/KURE-Project.git",
            "/tmp/KURE-Project"
        ], check=True)
    
    os.chdir("/tmp/KURE-Project")
    
    # Install the package
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
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
    
    # Run sampling
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Sampling failed with error: {result.stderr}")
        raise RuntimeError(f"Sampling failed: {result.stderr}")
    
    print("Image generation completed successfully!")
    print(result.stdout)
    
    # List generated files
    generated_files = []
    if os.path.exists(sample_dir):
        generated_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    return {
        "status": "success",
        "output_dir": sample_dir,
        "generated_files": generated_files,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.function(
    image=image,
    gpu=modal.gpu.T4(),
    volumes={
        "/models": models_volume,
        "/datasets": datasets_volume,
        "/outputs": outputs_volume
    },
    timeout=1800  # 30 minutes timeout
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
    
    # Clone the repository if not exists
    if not os.path.exists("/tmp/KURE-Project"):
        subprocess.run([
            "git", "clone", 
            "https://github.com/Todd7777/KURE-Project.git",
            "/tmp/KURE-Project"
        ], check=True)
    
    os.chdir("/tmp/KURE-Project")
    
    # Install the package
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Prepare evaluation command
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", checkpoint_path,
        "--dataset", dataset,
        "--steps"] + [str(s) for s in steps] + [
        "--metrics"] + metrics
    
    print(f"Running evaluation command: {' '.join(cmd)}")
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Evaluation failed with error: {result.stderr}")
        raise RuntimeError(f"Evaluation failed: {result.stderr}")
    
    print("Evaluation completed successfully!")
    print(result.stdout)
    
    return {
        "status": "success",
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.function(
    image=image,
    timeout=300
)
def list_available_configs():
    """List available configuration files"""
    import subprocess
    import os
    
    # Clone the repository if not exists
    if not os.path.exists("/tmp/KURE-Project"):
        subprocess.run([
            "git", "clone", 
            "https://github.com/Todd7777/KURE-Project.git",
            "/tmp/KURE-Project"
        ], check=True)
    
    configs_dir = "/tmp/KURE-Project/configs"
    if os.path.exists(configs_dir):
        configs = [f for f in os.listdir(configs_dir) if f.endswith('.yaml')]
        return {"available_configs": configs}
    else:
        return {"available_configs": []}

# Web interface for easy access
@app.function(
    image=image,
    allow_concurrent_inputs=10
)
@modal.web_endpoint(method="GET")
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
        #     wandb_project="kure-nrf-demo"
        # )
        # print("Training result:", result)
        
        # Example image generation (uncomment to run)
        # result = generate_images.remote(
        #     prompt="A red cube on top of a blue sphere",
        #     steps=4,
        #     num_samples=8
        # )
        # print("Generation result:", result)
