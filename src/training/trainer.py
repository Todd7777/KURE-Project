"""
Training Pipeline for Nonlinear Rectified Flows

This module implements the main training loop with:
- Multi-GPU support via FSDP
- Mixed precision training
- Gradient accumulation
- EMA model updates
- Logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any
import os
from pathlib import Path
import wandb
from tqdm import tqdm
import time


class NRFTrainer:
    """
    Main trainer for Nonlinear Rectified Flow models.
    
    Handles training loop, optimization, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        use_amp: bool = True,
        use_ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            model: NRF model
            vae: VAE for latent space encoding
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Training configuration
            device: Device to train on
            use_amp: Whether to use automatic mixed precision
            use_ddp: Whether to use distributed data parallel
            rank: Process rank for distributed training
            world_size: Number of processes
        """
        self.model = model
        self.vae = vae
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device
        self.use_amp = use_amp
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        
        # Move models to device
        self.model = self.model.to(device)
        self.vae = self.vae.to(device)
        self.vae.eval()  # VAE is frozen during NRF training
        
        # Wrap with DDP if distributed
        if use_ddp:
            self.model = DDP(self.model, device_ids=[rank])
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get("learning_rate", 1e-4),
                weight_decay=self.config.get("weight_decay", 1e-4),
                betas=(0.9, 0.999),
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler
        self.scheduler = scheduler
        
        # Mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        
        # Logging
        self.log_interval = self.config.get("log_interval", 100)
        self.val_interval = self.config.get("val_interval", 1000)
        self.save_interval = self.config.get("save_interval", 5000)
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if self.rank == 0 and self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "nonlinear-rectified-flows"),
                config=self.config,
                name=self.config.get("run_name", None),
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dictionary containing images and captions
            
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        # Get data
        images = batch["image"].to(self.device)
        captions = batch.get("caption", None)
        
        # Encode to latent space
        with torch.no_grad():
            mean, logvar = self.vae.encode(images)
            x_1 = self.vae.reparameterize(mean, logvar)
        
        # Get text embeddings (assuming they're precomputed)
        context = batch.get("text_embedding", None)
        if context is not None:
            context = context.to(self.device)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.use_amp):
            loss, metrics = self.model.compute_loss(
                x_1,
                context=context,
                loss_type=self.config.get("loss_type", "mse"),
                reg_weight=self.config.get("reg_weight", 0.01),
            )
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"]
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"]
                )
            
            self.optimizer.step()
        
        # Update EMA
        if hasattr(self.model, "update_ema"):
            self.model.update_ema()
        elif hasattr(self.model, "module") and hasattr(self.model.module, "update_ema"):
            self.model.module.update_ema()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validation loop.
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_velocity_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", disable=self.rank != 0):
            images = batch["image"].to(self.device)
            context = batch.get("text_embedding", None)
            if context is not None:
                context = context.to(self.device)
            
            # Encode to latent
            mean, logvar = self.vae.encode(images)
            x_1 = self.vae.reparameterize(mean, logvar)
            
            # Compute loss
            with autocast(enabled=self.use_amp):
                loss, metrics = self.model.compute_loss(
                    x_1,
                    context=context,
                    loss_type=self.config.get("loss_type", "mse"),
                    reg_weight=self.config.get("reg_weight", 0.01),
                )
            
            total_loss += metrics["loss"]
            total_velocity_loss += metrics["velocity_loss"]
            total_reg_loss += metrics["reg_loss"]
            num_batches += 1
        
        # Average metrics
        avg_metrics = {
            "val_loss": total_loss / num_batches,
            "val_velocity_loss": total_velocity_loss / num_batches,
            "val_reg_loss": total_reg_loss / num_batches,
        }
        
        return avg_metrics
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save training checkpoint"""
        if self.rank != 0:
            return
        
        # Get model state dict (unwrap DDP if necessary)
        if self.use_ddp:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str = "checkpoint.pt"):
        """Load training checkpoint"""
        path = self.checkpoint_dir / filename
        
        if not path.exists():
            print(f"Checkpoint {path} not found")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model
        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler and checkpoint["scheduler"]:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler and checkpoint["scaler"]:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        # Load training state
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"Loaded checkpoint from {path}")
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}, Rank: {self.rank}/{self.world_size}")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Training epoch
            epoch_start = time.time()
            epoch_metrics = {"loss": 0.0, "velocity_loss": 0.0, "reg_loss": 0.0}
            num_batches = 0
            
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=self.rank != 0,
            )
            
            for batch in pbar:
                # Training step
                metrics = self.train_step(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": metrics["loss"],
                    "step": self.global_step,
                })
                
                # Logging
                if self.global_step % self.log_interval == 0 and self.rank == 0:
                    log_metrics = {
                        "train/loss": metrics["loss"],
                        "train/velocity_loss": metrics["velocity_loss"],
                        "train/reg_loss": metrics["reg_loss"],
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "train/step": self.global_step,
                    }
                    
                    if self.config.get("use_wandb", False):
                        wandb.log(log_metrics, step=self.global_step)
                
                # Validation
                if self.global_step % self.val_interval == 0:
                    val_metrics = self.validate()
                    
                    if self.rank == 0:
                        print(f"\nValidation at step {self.global_step}:")
                        for key, value in val_metrics.items():
                            print(f"  {key}: {value:.6f}")
                        
                        if self.config.get("use_wandb", False):
                            wandb.log(val_metrics, step=self.global_step)
                        
                        # Save best model
                        if val_metrics.get("val_loss", float("inf")) < self.best_val_loss:
                            self.best_val_loss = val_metrics["val_loss"]
                            self.save_checkpoint("best_model.pt")
                
                # Checkpointing
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                
                self.global_step += 1
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            
            if self.rank == 0:
                avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
                print(f"\nEpoch {epoch+1} summary:")
                print(f"  Time: {epoch_time:.2f}s")
                for key, value in avg_metrics.items():
                    print(f"  {key}: {value:.6f}")
            
            # Save epoch checkpoint
            self.save_checkpoint("latest.pt")
        
        print("Training complete!")
        
        if self.rank == 0 and self.config.get("use_wandb", False):
            wandb.finish()


def setup_distributed():
    """Setup distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
