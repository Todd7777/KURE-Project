"""
Cubic Spline Teacher for Nonlinear Rectified Flow

This teacher uses cubic spline interpolation with learnable control points:
- A prompt-aware controller predicts K control points between z_0 and x_1
- Cubic splines provide smooth, twice-differentiable trajectories
- Regularization on path length and curvature keeps trajectories close to data manifold
"""

import torch
import torch.nn as nn
from typing import Optional, List
from ..nrf_base import TeacherBase


class CubicSplineController(nn.Module):
    """
    Prompt-aware controller that predicts spline control points.
    
    Given text embeddings and endpoints (z_0, x_1), this network predicts
    K intermediate control points that define a smooth cubic spline trajectory.
    """
    
    def __init__(
        self,
        context_dim: int = 768,
        latent_channels: int = 4,
        latent_size: int = 64,
        num_control_points: int = 3,
        hidden_dim: int = 512,
    ):
        """
        Args:
            context_dim: Dimension of text embeddings (e.g., CLIP)
            latent_channels: Number of channels in latent space
            latent_size: Spatial size of latent (e.g., 64x64)
            num_control_points: Number of intermediate control points
            hidden_dim: Hidden dimension for controller network
        """
        super().__init__()
        self.num_control_points = num_control_points
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        
        # Encode endpoints
        self.endpoint_encoder = nn.Sequential(
            nn.Conv2d(latent_channels * 2, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # Combine with text context
        self.context_fusion = nn.Sequential(
            nn.Linear(256 + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Predict control points
        # Each control point is a latent vector
        self.control_point_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, latent_channels * latent_size * latent_size),
            )
            for _ in range(num_control_points)
        ])
        
        # Initialize to produce small perturbations initially
        for module in self.control_point_predictor:
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
    
    def forward(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        context: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Predict control points.
        
        Args:
            z_0: Prior samples, shape (B, C, H, W)
            x_1: Data samples, shape (B, C, H, W)
            context: Text embeddings, shape (B, context_dim)
            
        Returns:
            control_points: List of K tensors, each shape (B, C, H, W)
        """
        batch_size = z_0.shape[0]
        
        # Encode endpoints
        endpoints = torch.cat([z_0, x_1], dim=1)
        endpoint_features = self.endpoint_encoder(endpoints)  # (B, 256)
        
        # Fuse with context
        combined = torch.cat([endpoint_features, context], dim=1)
        features = self.context_fusion(combined)  # (B, hidden_dim)
        
        # Predict control points
        control_points = []
        for predictor in self.control_point_predictor:
            cp = predictor(features)  # (B, C*H*W)
            cp = cp.view(batch_size, self.latent_channels, self.latent_size, self.latent_size)
            control_points.append(cp)
        
        return control_points


class CubicSplineTeacher(TeacherBase):
    """
    Cubic spline teacher with prompt-aware control points.
    
    The trajectory is defined by cubic spline interpolation through:
    - Start point: z_0
    - K control points: c_1, ..., c_K (predicted by controller)
    - End point: x_1
    
    We use natural cubic splines with continuous second derivatives.
    """
    
    def __init__(
        self,
        controller: CubicSplineController,
        path_length_weight: float = 0.1,
        curvature_weight: float = 0.05,
        control_point_weight: float = 0.01,
    ):
        """
        Args:
            controller: Network that predicts control points
            path_length_weight: Weight for path length regularization
            curvature_weight: Weight for curvature regularization
            control_point_weight: Weight for control point magnitude regularization
        """
        super().__init__()
        self.controller = controller
        self.path_length_weight = path_length_weight
        self.curvature_weight = curvature_weight
        self.control_point_weight = control_point_weight
        
        # Cache control points for efficiency
        self._cached_control_points = None
        self._cache_key = None
    
    def get_control_points(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        context: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Get or compute control points with caching"""
        # Simple cache based on batch size (for efficiency during training)
        cache_key = (z_0.shape[0], id(context))
        
        if self._cache_key == cache_key and self._cached_control_points is not None:
            return self._cached_control_points
        
        control_points = self.controller(z_0, x_1, context)
        self._cached_control_points = control_points
        self._cache_key = cache_key
        
        return control_points
    
    def _cubic_spline_basis(self, t: torch.Tensor, num_segments: int) -> tuple:
        """
        Compute cubic spline basis functions.
        
        For K control points, we have K+1 segments.
        We use Catmull-Rom splines for simplicity.
        
        Args:
            t: Time in [0, 1], shape (B,) or (B, 1, 1, 1)
            num_segments: Number of spline segments (K+1)
            
        Returns:
            segment_idx: Which segment t falls into
            local_t: Local time within segment [0, 1]
        """
        if t.dim() > 1:
            t = t.squeeze()
        
        # Map t to segment index
        t_scaled = t * num_segments
        segment_idx = torch.clamp(t_scaled.long(), 0, num_segments - 1)
        local_t = t_scaled - segment_idx.float()
        
        return segment_idx, local_t
    
    def interpolate(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cubic spline interpolation through control points.
        
        Args:
            z_0: Prior samples, shape (B, C, H, W)
            x_1: Data samples, shape (B, C, H, W)
            t: Time, shape (B,) or (B, 1, 1, 1)
            context: Text embeddings (required)
            
        Returns:
            x_t: Interpolated samples
        """
        if context is None:
            # Fall back to linear interpolation if no context
            if t.dim() == 1:
                t = t.view(-1, 1, 1, 1)
            return (1 - t) * z_0 + t * x_1
        
        # Get control points
        control_points = self.get_control_points(z_0, x_1, context)
        K = len(control_points)
        
        # All points: [z_0, c_1, ..., c_K, x_1]
        all_points = [z_0] + control_points + [x_1]
        num_segments = K + 1
        
        # Determine which segment each sample falls into
        if t.dim() == 1:
            t_scalar = t
        else:
            t_scalar = t.squeeze()
        
        segment_idx, local_t = self._cubic_spline_basis(t_scalar, num_segments)
        
        # Catmull-Rom spline interpolation
        # For each sample, interpolate within its segment
        batch_size = z_0.shape[0]
        x_t = torch.zeros_like(z_0)
        
        for i in range(batch_size):
            seg = segment_idx[i].item()
            tau = local_t[i].item()
            
            # Get 4 control points for Catmull-Rom (with boundary handling)
            p0 = all_points[max(0, seg - 1)][i]
            p1 = all_points[seg][i]
            p2 = all_points[min(len(all_points) - 1, seg + 1)][i]
            p3 = all_points[min(len(all_points) - 1, seg + 2)][i]
            
            # Catmull-Rom basis functions
            tau2 = tau * tau
            tau3 = tau2 * tau
            
            # Standard Catmull-Rom matrix
            x_t[i] = (
                0.5 * (
                    (-tau3 + 2 * tau2 - tau) * p0 +
                    (3 * tau3 - 5 * tau2 + 2) * p1 +
                    (-3 * tau3 + 4 * tau2 + tau) * p2 +
                    (tau3 - tau2) * p3
                )
            )
        
        return x_t
    
    def velocity(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute velocity as derivative of spline.
        
        For Catmull-Rom splines, the derivative is:
        v(τ) = 0.5 * [(-3τ² + 4τ - 1)p0 + (9τ² - 10τ)p1 + (-9τ² + 8τ + 1)p2 + (3τ² - 2τ)p3]
        
        Args:
            z_0: Prior samples
            x_1: Data samples
            t: Time
            context: Text embeddings
            
        Returns:
            v_t: Velocity at time t
        """
        if context is None:
            # Fall back to constant velocity
            return x_1 - z_0
        
        # Get control points
        control_points = self.get_control_points(z_0, x_1, context)
        K = len(control_points)
        all_points = [z_0] + control_points + [x_1]
        num_segments = K + 1
        
        if t.dim() == 1:
            t_scalar = t
        else:
            t_scalar = t.squeeze()
        
        segment_idx, local_t = self._cubic_spline_basis(t_scalar, num_segments)
        
        batch_size = z_0.shape[0]
        v_t = torch.zeros_like(z_0)
        
        for i in range(batch_size):
            seg = segment_idx[i].item()
            tau = local_t[i].item()
            
            # Get 4 control points
            p0 = all_points[max(0, seg - 1)][i]
            p1 = all_points[seg][i]
            p2 = all_points[min(len(all_points) - 1, seg + 1)][i]
            p3 = all_points[min(len(all_points) - 1, seg + 2)][i]
            
            # Derivative of Catmull-Rom spline
            tau2 = tau * tau
            
            # Note: Need to scale by num_segments because dt_local = num_segments * dt_global
            v_t[i] = num_segments * 0.5 * (
                (-3 * tau2 + 4 * tau - 1) * p0 +
                (9 * tau2 - 10 * tau) * p1 +
                (-9 * tau2 + 8 * tau + 1) * p2 +
                (3 * tau2 - 2 * tau) * p3
            )
        
        return v_t
    
    def regularization_loss(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Regularization on path length, curvature, and control point magnitude.
        
        Returns:
            Total regularization loss
        """
        if context is None:
            return torch.tensor(0.0, device=z_0.device)
        
        control_points = self.get_control_points(z_0, x_1, context)
        
        # All points for computing path length and curvature
        all_points = [z_0] + control_points + [x_1]
        
        # Path length regularization (sum of segment lengths)
        path_length = 0.0
        for i in range(len(all_points) - 1):
            segment_length = torch.mean((all_points[i + 1] - all_points[i]) ** 2)
            path_length += segment_length
        
        # Curvature regularization (second derivative)
        curvature = 0.0
        for i in range(1, len(all_points) - 1):
            # Discrete second derivative
            second_deriv = all_points[i - 1] - 2 * all_points[i] + all_points[i + 1]
            curvature += torch.mean(second_deriv ** 2)
        
        # Control point magnitude regularization
        cp_magnitude = sum(torch.mean(cp ** 2) for cp in control_points)
        
        total_reg = (
            self.path_length_weight * path_length +
            self.curvature_weight * curvature +
            self.control_point_weight * cp_magnitude
        )
        
        return total_reg
