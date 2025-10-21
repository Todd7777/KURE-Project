"""
Core Nonlinear Rectified Flow (NRF) Framework

This module implements the base NRF class that supports multiple teacher types
and generalized time scheduling for geometry-aware flow matching.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict, Any
from abc import ABC, abstractmethod
import math


class TimeScheduler:
    """
    Generalized time scheduling: ẋ(t) = a(t)·(data-pull) + b(t)·(prior-push)
    
    Following Luo et al. (2025), this allows flexible interpolation schedules
    that recover linear RF as a special case and enable nonlinear regimes.
    """
    
    def __init__(self, schedule_type: str = "linear", **kwargs):
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        
    def a(self, t: torch.Tensor) -> torch.Tensor:
        """Data-pull coefficient a(t)"""
        if self.schedule_type == "linear":
            return torch.ones_like(t)
        elif self.schedule_type == "cosine":
            return torch.cos(math.pi / 2 * t)
        elif self.schedule_type == "sigmoid":
            beta = self.kwargs.get("beta", 5.0)
            return torch.sigmoid(beta * (1 - t))
        elif self.schedule_type == "exponential":
            alpha = self.kwargs.get("alpha", 2.0)
            return torch.exp(-alpha * t)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def b(self, t: torch.Tensor) -> torch.Tensor:
        """Prior-push coefficient b(t)"""
        if self.schedule_type == "linear":
            return torch.ones_like(t)
        elif self.schedule_type == "cosine":
            return torch.sin(math.pi / 2 * t)
        elif self.schedule_type == "sigmoid":
            beta = self.kwargs.get("beta", 5.0)
            return torch.sigmoid(beta * t)
        elif self.schedule_type == "exponential":
            alpha = self.kwargs.get("alpha", 2.0)
            return torch.exp(-alpha * (1 - t))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class TeacherBase(ABC):
    """
    Abstract base class for NRF teachers.
    
    A teacher defines the supervision signal for training the velocity field.
    Different teachers implement different interpolation strategies between
    the prior z_0 ~ N(0, I) and data x_1.
    """
    
    @abstractmethod
    def interpolate(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute interpolated point x_t at time t.
        
        Args:
            z_0: Prior samples, shape (B, C, H, W)
            x_1: Data samples, shape (B, C, H, W)
            t: Time, shape (B,) or (B, 1, 1, 1)
            context: Optional conditioning (e.g., text embeddings)
            
        Returns:
            x_t: Interpolated samples, shape (B, C, H, W)
        """
        pass
    
    @abstractmethod
    def velocity(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute target velocity v_t = dx/dt at time t.
        
        Args:
            z_0: Prior samples, shape (B, C, H, W)
            x_1: Data samples, shape (B, C, H, W)
            t: Time, shape (B,) or (B, 1, 1, 1)
            context: Optional conditioning
            
        Returns:
            v_t: Target velocity, shape (B, C, H, W)
        """
        pass
    
    def regularization_loss(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Optional regularization loss (e.g., path length, curvature).
        
        Returns:
            Scalar regularization loss
        """
        return torch.tensor(0.0, device=z_0.device)


class NonlinearRectifiedFlow(nn.Module):
    """
    Nonlinear Rectified Flow (NRF) model.
    
    This is the main class that combines:
    - A velocity predictor network (e.g., U-Net)
    - A teacher for generating supervision signals
    - A time scheduler for flexible interpolation
    - Training and sampling procedures
    """
    
    def __init__(
        self,
        velocity_net: nn.Module,
        teacher: TeacherBase,
        time_scheduler: Optional[TimeScheduler] = None,
        ema_decay: float = 0.9999,
    ):
        """
        Args:
            velocity_net: Neural network that predicts velocity v_θ(x_t, t, c)
            teacher: Teacher model for generating supervision
            time_scheduler: Time scheduling strategy
            ema_decay: Exponential moving average decay for model weights
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.teacher = teacher
        self.time_scheduler = time_scheduler or TimeScheduler("linear")
        
        # EMA model for stable sampling
        self.ema_decay = ema_decay
        self.ema_velocity_net = None
        if ema_decay > 0:
            self.ema_velocity_net = self._create_ema_model()
    
    def _create_ema_model(self) -> nn.Module:
        """Create EMA copy of velocity network"""
        import copy
        ema_model = copy.deepcopy(self.velocity_net)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def update_ema(self):
        """Update EMA model weights"""
        if self.ema_velocity_net is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_velocity_net.parameters(),
                self.velocity_net.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        Predict velocity at (x_t, t).
        
        Args:
            x_t: Current state, shape (B, C, H, W)
            t: Time, shape (B,)
            context: Conditioning (e.g., text embeddings)
            use_ema: Whether to use EMA model
            
        Returns:
            v_θ(x_t, t, c): Predicted velocity
        """
        # Ensure t has correct shape for broadcasting
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        
        net = self.ema_velocity_net if (use_ema and self.ema_velocity_net) else self.velocity_net
        return net(x_t, t, context)
    
    def compute_loss(
        self,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        loss_type: str = "mse",
        reg_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.
        
        Args:
            x_1: Data samples, shape (B, C, H, W)
            context: Conditioning
            loss_type: Loss function type ("mse", "huber", "lpips")
            reg_weight: Weight for regularization loss
            
        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        batch_size = x_1.shape[0]
        device = x_1.device
        
        # Sample prior z_0 ~ N(0, I)
        z_0 = torch.randn_like(x_1)
        
        # Sample time uniformly t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Get interpolated point and target velocity from teacher
        x_t = self.teacher.interpolate(z_0, x_1, t, context)
        v_target = self.teacher.velocity(z_0, x_1, t, context)
        
        # Predict velocity
        v_pred = self.forward(x_t, t, context, use_ema=False)
        
        # Compute velocity matching loss
        if loss_type == "mse":
            velocity_loss = torch.mean((v_pred - v_target) ** 2)
        elif loss_type == "huber":
            velocity_loss = torch.nn.functional.huber_loss(v_pred, v_target)
        elif loss_type == "lpips":
            # For perceptual loss, requires LPIPS model
            raise NotImplementedError("LPIPS loss not yet implemented")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Add regularization from teacher
        reg_loss = self.teacher.regularization_loss(z_0, x_1, context)
        
        # Total loss
        total_loss = velocity_loss + reg_weight * reg_loss
        
        metrics = {
            "loss": total_loss.item(),
            "velocity_loss": velocity_loss.item(),
            "reg_loss": reg_loss.item(),
        }
        
        return total_loss, metrics
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        shape: Tuple[int, ...],
        context: Optional[torch.Tensor] = None,
        num_steps: int = 4,
        solver: str = "euler",
        use_ema: bool = True,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample from the model using ODE integration.
        
        Args:
            batch_size: Number of samples
            shape: Shape of samples (C, H, W)
            context: Conditioning
            num_steps: Number of integration steps
            solver: ODE solver ("euler", "heun", "rk4")
            use_ema: Whether to use EMA model
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            x_1: Generated samples
        """
        device = next(self.parameters()).device
        
        # Start from prior z_0 ~ N(0, I)
        x_t = torch.randn(batch_size, *shape, device=device)
        
        # Time discretization
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            
            # Predict velocity
            v_t = self.forward(x_t, t, context, use_ema=use_ema)
            
            # Classifier-free guidance
            if guidance_scale != 1.0 and context is not None:
                v_uncond = self.forward(x_t, t, None, use_ema=use_ema)
                v_t = v_uncond + guidance_scale * (v_t - v_uncond)
            
            # ODE integration step
            if solver == "euler":
                x_t = x_t + dt * v_t
            elif solver == "heun":
                # Heun's method (2nd order)
                x_temp = x_t + dt * v_t
                t_next = torch.full((batch_size,), (step + 1) * dt, device=device)
                v_next = self.forward(x_temp, t_next, context, use_ema=use_ema)
                x_t = x_t + dt * (v_t + v_next) / 2
            elif solver == "rk4":
                # Runge-Kutta 4th order
                k1 = v_t
                k2 = self.forward(x_t + dt * k1 / 2, t + dt / 2, context, use_ema=use_ema)
                k3 = self.forward(x_t + dt * k2 / 2, t + dt / 2, context, use_ema=use_ema)
                k4 = self.forward(x_t + dt * k3, t + dt, context, use_ema=use_ema)
                x_t = x_t + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            else:
                raise ValueError(f"Unknown solver: {solver}")
        
        return x_t
    
    def get_trajectory(
        self,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        num_points: int = 10,
    ) -> torch.Tensor:
        """
        Get the full trajectory from z_0 to x_1 according to the teacher.
        
        Useful for visualization and analysis.
        
        Args:
            x_1: Data samples
            context: Conditioning
            num_points: Number of points along trajectory
            
        Returns:
            trajectory: Shape (num_points, B, C, H, W)
        """
        z_0 = torch.randn_like(x_1)
        trajectory = []
        
        for i in range(num_points):
            t = torch.full((x_1.shape[0],), i / (num_points - 1), device=x_1.device)
            x_t = self.teacher.interpolate(z_0, x_1, t, context)
            trajectory.append(x_t)
        
        return torch.stack(trajectory, dim=0)
