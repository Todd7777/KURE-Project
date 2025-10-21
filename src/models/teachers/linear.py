"""
Linear Teacher for Rectified Flow

This is the baseline teacher that uses straight-line interpolation:
    x_t = t·x_1 + (1-t)·z_0
    v_t = x_1 - z_0

This is the standard RF formulation (Liu et al., 2022).
"""

import torch
from typing import Optional
from ..nrf_base import TeacherBase


class LinearTeacher(TeacherBase):
    """
    Linear (straight-line) teacher for rectified flow.
    
    This is the baseline that assumes constant velocity along straight paths.
    It's simple and stable but may drive trajectories through low-probability
    regions of the data manifold.
    """
    
    def __init__(self):
        super().__init__()
    
    def interpolate(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1-t)·z_0 + t·x_1
        
        Args:
            z_0: Prior samples, shape (B, C, H, W)
            x_1: Data samples, shape (B, C, H, W)
            t: Time, shape (B,) or (B, 1, 1, 1)
            context: Unused for linear teacher
            
        Returns:
            x_t: Interpolated samples
        """
        # Ensure t has correct shape for broadcasting
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        
        return (1 - t) * z_0 + t * x_1
    
    def velocity(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Constant velocity: v_t = x_1 - z_0
        
        The velocity is independent of time for linear interpolation.
        
        Args:
            z_0: Prior samples
            x_1: Data samples
            t: Time (unused for linear teacher)
            context: Unused
            
        Returns:
            v_t: Target velocity (constant)
        """
        return x_1 - z_0
    
    def regularization_loss(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """No regularization for linear teacher"""
        return torch.tensor(0.0, device=z_0.device)
