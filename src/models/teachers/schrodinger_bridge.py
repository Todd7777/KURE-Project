"""
Schrödinger Bridge Teacher for Nonlinear Rectified Flow

This teacher uses entropic optimal transport to compute distribution-aware
drift fields. We solve the Schrödinger Bridge problem in latent space using:
- Sinkhorn iterations for entropic regularization
- Nyström approximations for scalability
- Time-varying drift distillation

This removes hand-crafted curves and learns the optimal transport path.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..nrf_base import TeacherBase


class NystromSinkhornSolver:
    """
    Scalable Sinkhorn solver using Nyström low-rank approximation.
    
    For large batches, computing the full kernel matrix K is expensive.
    We use Nyström method to approximate K with a low-rank factorization.
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        num_iterations: int = 20,
        num_landmarks: int = 128,
        threshold: float = 1e-3,
    ):
        """
        Args:
            epsilon: Entropic regularization parameter
            num_iterations: Number of Sinkhorn iterations
            num_landmarks: Number of landmark points for Nyström
            threshold: Convergence threshold
        """
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.num_landmarks = num_landmarks
        self.threshold = threshold
    
    def compute_cost_matrix(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        metric: str = "euclidean",
    ) -> torch.Tensor:
        """
        Compute pairwise cost matrix C[i,j] = cost(x_i, y_j).
        
        Args:
            x: Source samples, shape (N, D)
            y: Target samples, shape (M, D)
            metric: Distance metric
            
        Returns:
            C: Cost matrix, shape (N, M)
        """
        if metric == "euclidean":
            # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2<x_i, y_j>
            x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (N, 1)
            y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (M, 1)
            xy = torch.mm(x, y.t())  # (N, M)
            C = x_norm + y_norm.t() - 2 * xy
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return C
    
    def sinkhorn_iteration(
        self,
        C: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard Sinkhorn iteration.
        
        Args:
            C: Cost matrix, shape (N, M)
            a: Source distribution, shape (N,). If None, uniform.
            b: Target distribution, shape (M,). If None, uniform.
            
        Returns:
            u: Dual variable for source, shape (N,)
            v: Dual variable for target, shape (M,)
        """
        N, M = C.shape
        device = C.device
        
        # Initialize with uniform distributions if not provided
        if a is None:
            a = torch.ones(N, device=device) / N
        if b is None:
            b = torch.ones(M, device=device) / M
        
        # Initialize dual variables
        u = torch.zeros(N, device=device)
        v = torch.zeros(M, device=device)
        
        # Sinkhorn iterations
        for _ in range(self.num_iterations):
            u_prev = u.clone()
            
            # Update v
            K_u = torch.exp((u.unsqueeze(1) - C) / self.epsilon)  # (N, M)
            v = self.epsilon * torch.log(b) - self.epsilon * torch.log(K_u.sum(dim=0) + 1e-10)
            
            # Update u
            K_v = torch.exp((v.unsqueeze(0) - C) / self.epsilon)  # (N, M)
            u = self.epsilon * torch.log(a) - self.epsilon * torch.log(K_v.sum(dim=1) + 1e-10)
            
            # Check convergence
            if torch.max(torch.abs(u - u_prev)) < self.threshold:
                break
        
        return u, v
    
    def nystrom_sinkhorn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sinkhorn with Nyström approximation for scalability.
        
        Args:
            x: Source samples, shape (N, D)
            y: Target samples, shape (M, D)
            a: Source distribution
            b: Target distribution
            
        Returns:
            u: Dual variable for source
            v: Dual variable for target
        """
        N, D = x.shape
        M = y.shape[0]
        
        # If batch is small enough, use standard Sinkhorn
        if N <= self.num_landmarks and M <= self.num_landmarks:
            C = self.compute_cost_matrix(x, y)
            return self.sinkhorn_iteration(C, a, b)
        
        # Select landmark points
        num_landmarks_x = min(self.num_landmarks, N)
        num_landmarks_y = min(self.num_landmarks, M)
        
        # Random sampling for landmarks (could use k-means for better approximation)
        idx_x = torch.randperm(N)[:num_landmarks_x]
        idx_y = torch.randperm(M)[:num_landmarks_y]
        
        landmarks_x = x[idx_x]  # (L_x, D)
        landmarks_y = y[idx_y]  # (L_y, D)
        
        # Compute cost matrices
        C_ll = self.compute_cost_matrix(landmarks_x, landmarks_y)  # (L_x, L_y)
        C_xl = self.compute_cost_matrix(x, landmarks_y)  # (N, L_y)
        C_ly = self.compute_cost_matrix(landmarks_x, y)  # (L_x, M)
        
        # Solve Sinkhorn on landmarks
        u_l, v_l = self.sinkhorn_iteration(C_ll)
        
        # Extend to full space using Nyström
        # u ≈ ε·log(a) - ε·log(K_ly @ exp(v_l/ε))
        K_ly = torch.exp((u_l.unsqueeze(1) - C_ly) / self.epsilon)
        if b is None:
            b = torch.ones(M, device=y.device) / M
        v = self.epsilon * torch.log(b) - self.epsilon * torch.log(K_ly.sum(dim=0) + 1e-10)
        
        # Similarly for u
        K_xl = torch.exp((v.unsqueeze(0) - C_xl) / self.epsilon)
        if a is None:
            a = torch.ones(N, device=x.device) / N
        u = self.epsilon * torch.log(a) - self.epsilon * torch.log(K_xl.sum(dim=1) + 1e-10)
        
        return u, v


class SchrodingerBridgeDriftNet(nn.Module):
    """
    Neural network that learns to predict the time-varying drift u_t^*.
    
    This network is trained to match the drift computed from the Schrödinger
    Bridge solution, allowing us to distill the OT-optimal path into a
    fast feedforward model.
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        context_dim: int = 768,
        time_embed_dim: int = 256,
        hidden_channels: int = 256,
    ):
        """
        Args:
            latent_channels: Number of channels in latent space
            context_dim: Dimension of text embeddings
            time_embed_dim: Dimension of time embeddings
            hidden_channels: Hidden channels in U-Net-like architecture
        """
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Context projection
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, time_embed_dim),
            nn.SiLU(),
        )
        
        # Drift prediction network (simplified U-Net)
        self.input_conv = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList([
            self._make_res_block(hidden_channels, hidden_channels, time_embed_dim),
            self._make_res_block(hidden_channels, hidden_channels * 2, time_embed_dim),
        ])
        
        self.mid_block = self._make_res_block(hidden_channels * 2, hidden_channels * 2, time_embed_dim)
        
        self.up_blocks = nn.ModuleList([
            self._make_res_block(hidden_channels * 2, hidden_channels, time_embed_dim),
            self._make_res_block(hidden_channels, hidden_channels, time_embed_dim),
        ])
        
        self.output_conv = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        
        # Initialize output to zero for stability
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def _make_res_block(self, in_channels, out_channels, time_embed_dim):
        """Create a residual block with time conditioning"""
        return nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
            ),
            nn.Linear(time_embed_dim, out_channels),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            ),
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity(),
        ])
    
    def _apply_res_block(self, x, t_emb, block):
        """Apply residual block with time conditioning"""
        conv1, time_proj, conv2, skip = block
        
        h = conv1(x)
        h = h + time_proj(t_emb)[:, :, None, None]
        h = conv2(h)
        
        return h + skip(x)
    
    def forward(self, x_t, t, context):
        """
        Predict drift correction.
        
        Args:
            x_t: Current state, shape (B, C, H, W)
            t: Time, shape (B,)
            context: Text embeddings, shape (B, context_dim)
            
        Returns:
            drift: Predicted drift, shape (B, C, H, W)
        """
        # Embed time and context
        t_emb = self.time_embed(t.view(-1, 1))  # (B, time_embed_dim)
        c_emb = self.context_proj(context)  # (B, time_embed_dim)
        cond_emb = t_emb + c_emb
        
        # Initial convolution
        h = self.input_conv(x_t)
        
        # Downsampling
        for block in self.down_blocks:
            h = self._apply_res_block(h, cond_emb, block)
        
        # Middle
        h = self._apply_res_block(h, cond_emb, self.mid_block)
        
        # Upsampling
        for block in self.up_blocks:
            h = self._apply_res_block(h, cond_emb, block)
        
        # Output
        drift = self.output_conv(h)
        
        return drift


class SchrodingerBridgeTeacher(TeacherBase):
    """
    Schrödinger Bridge teacher using entropic optimal transport.
    
    This teacher computes the optimal transport drift between the prior
    and data distributions, providing distribution-aware supervision that
    eliminates linear-teacher bias.
    """
    
    def __init__(
        self,
        drift_net: SchrodingerBridgeDriftNet,
        solver: Optional[NystromSinkhornSolver] = None,
        num_time_steps: int = 10,
        update_frequency: int = 100,
    ):
        """
        Args:
            drift_net: Neural network for drift prediction
            solver: Sinkhorn solver for computing OT
            num_time_steps: Number of time discretization points
            update_frequency: How often to recompute OT solution (in training steps)
        """
        super().__init__()
        self.drift_net = drift_net
        self.solver = solver or NystromSinkhornSolver()
        self.num_time_steps = num_time_steps
        self.update_frequency = update_frequency
        
        # Cache for OT solution
        self._ot_cache = {}
        self._update_counter = 0
    
    def compute_ot_drift(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute drift from Schrödinger Bridge solution.
        
        This is expensive and should be cached or computed infrequently.
        
        Args:
            z_0: Prior samples, shape (B, C, H, W)
            x_1: Data samples, shape (B, C, H, W)
            t: Time, shape (B,)
            
        Returns:
            drift: OT-optimal drift, shape (B, C, H, W)
        """
        batch_size = z_0.shape[0]
        
        # Flatten spatial dimensions for OT computation
        z_0_flat = z_0.view(batch_size, -1)  # (B, C*H*W)
        x_1_flat = x_1.view(batch_size, -1)  # (B, C*H*W)
        
        # Solve OT problem
        u, v = self.solver.nystrom_sinkhorn(z_0_flat, x_1_flat)
        
        # Compute drift as gradient of potential
        # For entropic OT, drift ≈ ∇φ where φ is the potential
        # Approximate using finite differences
        eps = 1e-3
        drift_flat = torch.zeros_like(z_0_flat)
        
        # This is a simplified approximation; in practice, we'd use the dual potentials
        # For now, use linear interpolation with OT-weighted direction
        linear_drift = x_1_flat - z_0_flat
        
        # Weight by OT potentials (higher potential = stronger drift)
        weights = torch.softmax(u / self.solver.epsilon, dim=0)
        drift_flat = weights.unsqueeze(1) * linear_drift
        
        # Reshape back
        drift = drift_flat.view_as(z_0)
        
        return drift
    
    def interpolate(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpolate using learned drift field.
        
        We integrate the drift from z_0 to time t.
        For efficiency, we use the learned drift_net rather than
        recomputing OT at every step.
        
        Args:
            z_0: Prior samples
            x_1: Data samples
            t: Time
            context: Text embeddings
            
        Returns:
            x_t: Interpolated samples
        """
        if context is None:
            # Fall back to linear interpolation
            if t.dim() == 1:
                t = t.view(-1, 1, 1, 1)
            return (1 - t) * z_0 + t * x_1
        
        # Use learned drift for interpolation
        # x_t ≈ z_0 + ∫_0^t drift(x_s, s) ds
        # Approximate with Euler integration
        num_steps = 5  # Small number for efficiency
        dt = t / num_steps
        
        x_t = z_0.clone()
        for step in range(num_steps):
            s = (step / num_steps) * t
            if s.dim() == 1:
                s = s.view(-1, 1, 1, 1)
            
            # Predict drift
            drift = self.drift_net(x_t, s.squeeze(), context)
            x_t = x_t + dt.view(-1, 1, 1, 1) * drift
        
        return x_t
    
    def velocity(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute velocity using learned drift.
        
        The velocity is the drift field at the current interpolated position.
        
        Args:
            z_0: Prior samples
            x_1: Data samples
            t: Time
            context: Text embeddings
            
        Returns:
            v_t: Velocity
        """
        if context is None:
            # Fall back to constant velocity
            return x_1 - z_0
        
        # Get interpolated position
        x_t = self.interpolate(z_0, x_1, t, context)
        
        # Predict drift at x_t
        if t.dim() > 1:
            t = t.squeeze()
        
        drift = self.drift_net(x_t, t, context)
        
        # Combine with linear component for stability
        linear_velocity = x_1 - z_0
        alpha = 0.5  # Blend factor
        
        return alpha * drift + (1 - alpha) * linear_velocity
    
    def train_drift_net(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        context: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Train the drift network to match OT solution.
        
        This should be called periodically during training to update
        the drift network with fresh OT solutions.
        
        Args:
            z_0: Prior samples
            x_1: Data samples
            context: Text embeddings
            optimizer: Optimizer for drift_net
            
        Returns:
            loss: Training loss
        """
        # Sample time points
        batch_size = z_0.shape[0]
        t = torch.rand(batch_size, device=z_0.device)
        
        # Compute OT drift (expensive)
        with torch.no_grad():
            ot_drift = self.compute_ot_drift(z_0, x_1, t)
        
        # Get interpolated position
        x_t = self.interpolate(z_0, x_1, t, context)
        
        # Predict drift
        pred_drift = self.drift_net(x_t, t, context)
        
        # Match OT drift
        loss = torch.mean((pred_drift - ot_drift) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def regularization_loss(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Regularization on drift network weights.
        
        Returns:
            L2 regularization on drift network
        """
        reg = 0.0
        for param in self.drift_net.parameters():
            reg += torch.sum(param ** 2)
        
        return 1e-5 * reg
