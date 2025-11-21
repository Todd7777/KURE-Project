"""
Main training script for Nonlinear Rectified Flows

Usage:
    python scripts/train.py --config configs/quadratic.yaml --gpus 4
"""

import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from models.nrf_base import NonlinearRectifiedFlow, TimeScheduler
from models.teachers.linear import LinearTeacher
from models.teachers.quadratic import QuadraticTeacher, AdaptiveQuadraticTeacher
from models.teachers.cubic_spline import CubicSplineTeacher, CubicSplineController
from models.teachers.schrodinger_bridge import (
    SchrodingerBridgeTeacher,
    SchrodingerBridgeDriftNet,
    NystromSinkhornSolver,
)
from models.unet import UNetVelocityPredictor
from models.vae import create_vae
from training.trainer import NRFTrainer, setup_distributed, cleanup_distributed
from data.datasets import create_dataloader


def load_config(config_path: str) -> dict:
    """Load YAML config with support for _base_ inheritance"""
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Handle base config inheritance
    if "_base_" in config:
        base_path = config_path.parent / config["_base_"]
        base_config = load_config(base_path)
        # Merge: base config first, then override with current config
        merged = {**base_config, **config}
        # Remove _base_ key
        merged.pop("_base_", None)
        return merged
    
    return config


def create_teacher(config: dict, device: str):
    """Create teacher model based on config"""
    teacher_type = config["teacher"]["type"]
    
    if teacher_type == "linear":
        return LinearTeacher()
    
    elif teacher_type == "quadratic":
        if config["teacher"].get("adaptive", False):
            return AdaptiveQuadraticTeacher(
                context_dim=config["model"]["context_dim"],
                alpha_min=config["teacher"].get("alpha_min", -0.5),
                alpha_max=config["teacher"].get("alpha_max", 1.0),
            )
        else:
            return QuadraticTeacher(
                alpha=config["teacher"].get("alpha", 0.5),
                learnable=config["teacher"].get("learnable", False),
                alpha_schedule=config["teacher"].get("alpha_schedule", None),
            )
    
    elif teacher_type == "cubic_spline":
        controller = CubicSplineController(
            context_dim=config["model"]["context_dim"],
            latent_channels=config["model"]["latent_channels"],
            latent_size=config["model"]["latent_size"],
            num_control_points=config["teacher"].get("num_control_points", 3),
            hidden_dim=config["teacher"].get("hidden_dim", 512),
        ).to(device)
        
        return CubicSplineTeacher(
            controller=controller,
            path_length_weight=config["teacher"].get("path_length_weight", 0.1),
            curvature_weight=config["teacher"].get("curvature_weight", 0.05),
            control_point_weight=config["teacher"].get("control_point_weight", 0.01),
        )
    
    elif teacher_type == "schrodinger_bridge":
        drift_net = SchrodingerBridgeDriftNet(
            latent_channels=config["model"]["latent_channels"],
            context_dim=config["model"]["context_dim"],
            time_embed_dim=config["teacher"].get("time_embed_dim", 256),
            hidden_channels=config["teacher"].get("hidden_channels", 256),
        ).to(device)
        
        solver = NystromSinkhornSolver(
            epsilon=config["teacher"].get("epsilon", 0.1),
            num_iterations=config["teacher"].get("num_iterations", 20),
            num_landmarks=config["teacher"].get("num_landmarks", 128),
        )
        
        return SchrodingerBridgeTeacher(
            drift_net=drift_net,
            solver=solver,
            num_time_steps=config["teacher"].get("num_time_steps", 10),
            update_frequency=config["teacher"].get("update_frequency", 100),
        )
    
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")


def main(args):
    """Main training function"""
    
    # Load config with inheritance support
    config = load_config(args.config)
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    print(f"Rank {rank}/{world_size}, Device: {device}")
    
    # Create VAE
    print("Creating VAE...")
    vae = create_vae(
        image_size=config["data"]["image_size"],
        latent_size=config["model"]["latent_size"],
        in_channels=config["data"]["in_channels"],
        latent_channels=config["model"]["latent_channels"],
    ).to(device)
    
    # Load pretrained VAE if available
    if config.get("vae_checkpoint"):
        vae.load_state_dict(torch.load(config["vae_checkpoint"], map_location=device))
        print(f"Loaded VAE from {config['vae_checkpoint']}")
    
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    # Create teacher
    print(f"Creating {config['teacher']['type']} teacher...")
    teacher = create_teacher(config, device)
    
    # Create time scheduler
    time_scheduler = TimeScheduler(
        schedule_type=config["model"].get("time_schedule", "linear"),
    )
    
    # Create velocity predictor
    print("Creating velocity predictor...")
    velocity_net = UNetVelocityPredictor(
        in_channels=config["model"]["latent_channels"],
        out_channels=config["model"]["latent_channels"],
        model_channels=config["model"]["model_channels"],
        channel_mult=tuple(config["model"]["channel_mult"]),
        num_res_blocks=config["model"]["num_res_blocks"],
        context_dim=config["model"]["context_dim"],
        time_embed_dim=config["model"]["time_embed_dim"],
    ).to(device)
    
    # Create NRF model
    print("Creating NRF model...")
    model = NonlinearRectifiedFlow(
        velocity_net=velocity_net,
        teacher=teacher,
        time_scheduler=time_scheduler,
        ema_decay=config["training"].get("ema_decay", 0.9999),
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloader(
        dataset_name=config["data"]["dataset"],
        split="train",
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=config["data"]["num_workers"],
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        dataset_name=config["data"]["dataset"],
        split="val",
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=config["data"]["num_workers"],
        shuffle=False,
    ) if config["data"].get("use_validation", True) else None
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=tuple(config["training"]["betas"]),
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"] * len(train_loader),
        eta_min=config["training"].get("min_lr", 1e-6),
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = NRFTrainer(
        model=model,
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config["training"],
        device=device,
        use_amp=config["training"].get("use_amp", True),
        use_ddp=world_size > 1,
        rank=rank,
        world_size=world_size,
    )
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("Starting training...")
    trainer.train(num_epochs=config["training"]["num_epochs"])
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nonlinear Rectified Flow")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    main(args)
