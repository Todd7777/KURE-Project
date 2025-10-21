"""
Quadratic Teacher for Nonlinear Rectified Flow

This teacher uses quadratic interpolation to allow modest curvature:
    x_t = (1-t)·z_0 + t·x_1 + α·t·(1-t)·(x_1 - z_0)
    
The quadratic term α·t·(1-t) provides the minimal curved departure from
constant velocity, allowing early-late semantic shifts without sacrificing
stability.
"""

import torch
import torch.nn as nn
from typing import Optional
from ..nrf_base import TeacherBase


class QuadraticTeacher(TeacherBase):
    """
    Quadratic teacher with learnable or fixed curvature parameter.
    
    The interpolation follows:
        x_t = (1-t)·z_0 + t·x_1 + α·t·(1-t)·Δ
    where Δ = x_1 - z_0 is the displacement vector.
    
    The parameter α controls the amount of curvature:
    - α = 0: Reduces to linear interpolation
    - α > 0: Bulges toward x_1 (accelerates early, decelerates late)
    - α < 0: Bulges toward z_0 (decelerates early, accelerates late)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        learnable: bool = False,
        alpha_schedule: Optional[str] = None,
    ):
        """
        Args:
            alpha: Curvature parameter
            learnable: Whether to learn alpha during training
            alpha_schedule: Optional time-dependent schedule for alpha
                          ("constant", "increasing", "decreasing")
        """
        super().__init__()
        self.alpha_schedule = alpha_schedule
        
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get alpha value, potentially time-dependent.
        
        Args:
            t: Time, shape (B,) or (B, 1, 1, 1)
            
        Returns:
            alpha_t: Curvature parameter at time t
        """
        alpha = self.alpha
        
        if self.alpha_schedule == "increasing":
            # More curvature later in the trajectory
            if t.dim() == 1:
                t_scalar = t
            else:
                t_scalar = t.squeeze()
            alpha = alpha * t_scalar.view(-1, 1, 1, 1)
        elif self.alpha_schedule == "decreasing":
            # More curvature early in the trajectory
            if t.dim() == 1:
                t_scalar = t
            else:
                t_scalar = t.squeeze()
            alpha = alpha * (1 - t_scalar).view(-1, 1, 1, 1)
        
        return alpha
    
    def interpolate(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Quadratic interpolation with curvature term.
        
        Args:
            z_0: Prior samples, shape (B, C, H, W)
            x_1: Data samples, shape (B, C, H, W)
            t: Time, shape (B,) or (B, 1, 1, 1)
            context: Unused for quadratic teacher
            
        Returns:
            x_t: Interpolated samples
        """
        # Ensure t has correct shape for broadcasting
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        
        # Get alpha (potentially time-dependent)
        alpha = self.get_alpha(t)
        
        # Displacement vector
        delta = x_1 - z_0
        
        # Quadratic interpolation
        # x_t = (1-t)·z_0 + t·x_1 + α·t·(1-t)·Δ
        linear_part = (1 - t) * z_0 + t * x_1
        quadratic_part = alpha * t * (1 - t) * delta
        
        return linear_part + quadratic_part
    
    def velocity(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute velocity v_t = dx/dt.
        
        For quadratic interpolation:
            x_t = (1-t)·z_0 + t·x_1 + α·t·(1-t)·Δ
        
        Taking derivative:
            v_t = -z_0 + x_1 + α·(1-2t)·Δ
                = Δ + α·(1-2t)·Δ
                = (1 + α·(1-2t))·Δ
        
        Args:
            z_0: Prior samples
            x_1: Data samples
            t: Time, shape (B,) or (B, 1, 1, 1)
            context: Unused
            
        Returns:
            v_t: Target velocity
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        
        # Get alpha
        alpha = self.get_alpha(t)
        
        # Displacement
        delta = x_1 - z_0
        
        # Velocity with quadratic correction
        # v_t = (1 + α·(1-2t))·Δ
        velocity_scale = 1 + alpha * (1 - 2 * t)
        
        return velocity_scale * delta
    
    def regularization_loss(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Optional regularization on alpha to prevent excessive curvature.
        
        Returns:
            L2 penalty on alpha
        """
        # Penalize large alpha values to maintain stability
        return 0.1 * (self.alpha ** 2)


class AdaptiveQuadraticTeacher(QuadraticTeacher):
    """
    Quadratic teacher with context-dependent curvature.
    
    This variant predicts alpha from the text embedding, allowing
    different prompts to have different trajectory curvatures.
    """
    
    def __init__(
        self,
        context_dim: int = 768,
        alpha_min: float = -0.5,
        alpha_max: float = 1.0,
    ):
        """
        Args:
            context_dim: Dimension of context embeddings (e.g., CLIP text)
            alpha_min: Minimum alpha value
            alpha_max: Maximum alpha value
        """
        super().__init__(alpha=0.5, learnable=False)
        
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Small MLP to predict alpha from context
        self.alpha_predictor = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def get_alpha(self, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict alpha from context.
        
        Args:
            t: Time (unused in this variant)
            context: Text embeddings, shape (B, context_dim)
            
        Returns:
            alpha: Context-dependent curvature, shape (B, 1, 1, 1)
        """
        if context is None:
            # Fall back to default alpha
            return self.alpha
        
        # Predict alpha in [0, 1] and scale to [alpha_min, alpha_max]
        alpha_normalized = self.alpha_predictor(context)  # (B, 1)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_normalized
        
        # Reshape for broadcasting
        return alpha.view(-1, 1, 1, 1)
    
    def interpolate(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Quadratic interpolation with context-dependent alpha"""
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        
        alpha = self.get_alpha(t, context)
        delta = x_1 - z_0
        
        linear_part = (1 - t) * z_0 + t * x_1
        quadratic_part = alpha * t * (1 - t) * delta
        
        return linear_part + quadratic_part
    
    def velocity(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Velocity with context-dependent curvature"""
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        
        alpha = self.get_alpha(t, context)
        delta = x_1 - z_0
        
        velocity_scale = 1 + alpha * (1 - 2 * t)
        return velocity_scale * delta
