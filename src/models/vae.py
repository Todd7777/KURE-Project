"""
VAE with Pullback Metrics for Geometry-Aware NRF

This module implements:
1. Standard VAE for latent space encoding/decoding
2. Pullback metric computation via decoder Jacobian
3. Geodesic distance and path length computation on the learned manifold
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.autograd as autograd


class VAEEncoder(nn.Module):
    """Encoder network for VAE"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 4),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        
        for mult in channel_mult:
            out_ch = base_channels * mult
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.GroupNorm(32, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(32, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
            ))
            ch = out_ch
        
        # Output projection to latent space
        self.output_conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, latent_channels * 2, 1),  # Mean and log-variance
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution.
        
        Args:
            x: Image, shape (B, C, H, W)
            
        Returns:
            mean: Latent mean, shape (B, latent_channels, H', W')
            logvar: Latent log-variance, shape (B, latent_channels, H', W')
        """
        h = self.input_conv(x)
        
        for block in self.down_blocks:
            h = block(h)
        
        h = self.output_conv(h)
        mean, logvar = h.chunk(2, dim=1)
        
        return mean, logvar


class VAEDecoder(nn.Module):
    """Decoder network for VAE"""
    
    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple = (4, 4, 2, 1),
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        
        # Input projection from latent space
        ch = base_channels * channel_mult[0]
        self.input_conv = nn.Sequential(
            nn.Conv2d(latent_channels, ch, 3, padding=1),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
        )
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * channel_mult[min(i + 1, len(channel_mult) - 1)]
            self.up_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
            ))
            ch = out_ch
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.
        
        Args:
            z: Latent code, shape (B, latent_channels, H', W')
            
        Returns:
            x: Reconstructed image, shape (B, out_channels, H, W)
        """
        h = self.input_conv(z)
        
        for block in self.up_blocks:
            h = block(h)
        
        x = self.output_conv(h)
        
        return x


class PullbackMetricVAE(nn.Module):
    """
    VAE with pullback metric computation.
    
    The pullback metric is defined as:
        G(z) = J_D(z)^T @ J_D(z)
    
    where J_D(z) is the Jacobian of the decoder at latent point z.
    
    This metric allows us to:
    1. Compute geodesic distances on the learned image manifold
    2. Regularize path lengths in latent space
    3. Measure curvature of trajectories
    """
    
    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        kl_weight: float = 1e-6,
    ):
        """
        Args:
            encoder: VAE encoder
            decoder: VAE decoder
            kl_weight: Weight for KL divergence loss
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image"""
        return self.decoder(z)
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input image
            
        Returns:
            recon: Reconstructed image
            mean: Latent mean
            logvar: Latent log-variance
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar
    
    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss.
        
        Args:
            x: Original image
            recon: Reconstructed image
            mean: Latent mean
            logvar: Latent log-variance
            
        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - recon) ** 2)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        metrics = {
            "vae_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        
        return total_loss, metrics
    
    def compute_jacobian(
        self,
        z: torch.Tensor,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute Jacobian of decoder J_D(z) = ∂D/∂z.
        
        This is expensive but necessary for pullback metric computation.
        
        Args:
            z: Latent code, shape (B, C, H, W)
            create_graph: Whether to create computation graph (for 2nd derivatives)
            
        Returns:
            jacobian: Shape (B, out_dim, latent_dim) where dimensions are flattened
        """
        batch_size = z.shape[0]
        z_flat = z.view(batch_size, -1)  # (B, latent_dim)
        latent_dim = z_flat.shape[1]
        
        # Decode
        z_input = z.requires_grad_(True)
        x = self.decode(z_input)
        x_flat = x.view(batch_size, -1)  # (B, out_dim)
        out_dim = x_flat.shape[1]
        
        # Compute Jacobian row by row
        jacobian = torch.zeros(batch_size, out_dim, latent_dim, device=z.device)
        
        for i in range(out_dim):
            # Compute gradient of i-th output w.r.t. z
            grad_outputs = torch.zeros_like(x_flat)
            grad_outputs[:, i] = 1.0
            
            grads = autograd.grad(
                outputs=x_flat,
                inputs=z_input,
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            
            jacobian[:, i, :] = grads.view(batch_size, -1)
        
        return jacobian
    
    def compute_pullback_metric(
        self,
        z: torch.Tensor,
        approximate: bool = True,
    ) -> torch.Tensor:
        """
        Compute pullback metric G(z) = J_D(z)^T @ J_D(z).
        
        Args:
            z: Latent code, shape (B, C, H, W)
            approximate: If True, use Monte Carlo approximation for efficiency
            
        Returns:
            metric: Pullback metric tensor, shape (B, latent_dim, latent_dim)
        """
        if approximate:
            # Monte Carlo approximation: G ≈ E[v^T J^T J v] for random v
            # This is much faster than computing full Jacobian
            return self._approximate_pullback_metric(z)
        else:
            # Exact computation (expensive)
            jacobian = self.compute_jacobian(z)  # (B, out_dim, latent_dim)
            metric = torch.bmm(jacobian.transpose(1, 2), jacobian)  # (B, latent_dim, latent_dim)
            return metric
    
    def _approximate_pullback_metric(
        self,
        z: torch.Tensor,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Approximate pullback metric using Hutchinson's trace estimator.
        
        G ≈ (1/num_samples) Σ (J^T J v) ⊗ v for random v ~ N(0, I)
        
        Args:
            z: Latent code
            num_samples: Number of random samples
            
        Returns:
            metric: Approximate pullback metric
        """
        batch_size = z.shape[0]
        z_flat = z.view(batch_size, -1)
        latent_dim = z_flat.shape[1]
        
        metric = torch.zeros(batch_size, latent_dim, latent_dim, device=z.device)
        
        for _ in range(num_samples):
            # Random direction
            v = torch.randn_like(z)
            
            # Compute J^T J v using double backward
            z_input = z.requires_grad_(True)
            x = self.decode(z_input)
            
            # First backward: J v
            jv = autograd.grad(
                outputs=x,
                inputs=z_input,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )[0]
            
            # Second backward: J^T (J v)
            jtjv = autograd.grad(
                outputs=jv,
                inputs=z_input,
                grad_outputs=torch.ones_like(jv),
                retain_graph=False,
            )[0]
            
            # Accumulate outer product
            v_flat = v.view(batch_size, -1, 1)
            jtjv_flat = jtjv.view(batch_size, -1, 1)
            metric += torch.bmm(jtjv_flat, v_flat.transpose(1, 2))
        
        metric /= num_samples
        
        return metric
    
    def geodesic_distance(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Compute approximate geodesic distance between two latent points.
        
        We approximate the geodesic by discretizing the straight line path
        and integrating the metric along it:
        
        d(z1, z2) ≈ Σ sqrt((z_{i+1} - z_i)^T G(z_i) (z_{i+1} - z_i))
        
        Args:
            z1: Start point, shape (B, C, H, W)
            z2: End point, shape (B, C, H, W)
            num_steps: Number of discretization steps
            
        Returns:
            distance: Geodesic distance, shape (B,)
        """
        batch_size = z1.shape[0]
        distance = torch.zeros(batch_size, device=z1.device)
        
        for i in range(num_steps):
            t = i / num_steps
            z_t = (1 - t) * z1 + t * z2
            
            # Tangent vector
            dz = (z2 - z1) / num_steps
            dz_flat = dz.view(batch_size, -1, 1)
            
            # Compute metric at z_t
            G = self.compute_pullback_metric(z_t, approximate=True)
            
            # Length element: sqrt(dz^T G dz)
            length = torch.sqrt(torch.bmm(
                torch.bmm(dz_flat.transpose(1, 2), G),
                dz_flat
            ).squeeze())
            
            distance += length
        
        return distance
    
    def path_length(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute path length of a trajectory in latent space.
        
        Args:
            trajectory: Shape (num_points, B, C, H, W)
            
        Returns:
            length: Total path length, shape (B,)
        """
        num_points = trajectory.shape[0]
        batch_size = trajectory.shape[1]
        length = torch.zeros(batch_size, device=trajectory.device)
        
        for i in range(num_points - 1):
            z1 = trajectory[i]
            z2 = trajectory[i + 1]
            length += self.geodesic_distance(z1, z2, num_steps=1)
        
        return length
    
    def curvature(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute curvature of a trajectory (discrete second derivative).
        
        Args:
            trajectory: Shape (num_points, B, C, H, W)
            
        Returns:
            curvature: Total curvature, shape (B,)
        """
        num_points = trajectory.shape[0]
        batch_size = trajectory.shape[1]
        curvature = torch.zeros(batch_size, device=trajectory.device)
        
        for i in range(1, num_points - 1):
            # Second derivative
            d2z = trajectory[i - 1] - 2 * trajectory[i] + trajectory[i + 1]
            curvature += torch.mean(d2z.view(batch_size, -1) ** 2, dim=1)
        
        return curvature


def create_vae(
    image_size: int = 256,
    latent_size: int = 64,
    in_channels: int = 3,
    latent_channels: int = 4,
) -> PullbackMetricVAE:
    """
    Factory function to create a VAE with pullback metrics.
    
    Args:
        image_size: Input image size
        latent_size: Latent spatial size
        in_channels: Number of input channels
        latent_channels: Number of latent channels
        
    Returns:
        vae: PullbackMetricVAE model
    """
    # Compute downsampling factor
    downsample_factor = image_size // latent_size
    num_downsample = int(torch.log2(torch.tensor(downsample_factor)).item())
    
    channel_mult = tuple([2 ** i for i in range(num_downsample)])
    
    encoder = VAEEncoder(
        in_channels=in_channels,
        latent_channels=latent_channels,
        channel_mult=channel_mult,
    )
    
    decoder = VAEDecoder(
        latent_channels=latent_channels,
        out_channels=in_channels,
        channel_mult=tuple(reversed(channel_mult)),
    )
    
    vae = PullbackMetricVAE(encoder, decoder)
    
    return vae
