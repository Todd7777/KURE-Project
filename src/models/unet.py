"""
U-Net Velocity Predictor for NRF

This implements a conditional U-Net that predicts the velocity field v_θ(x_t, t, c)
where:
- x_t is the current state
- t is the time
- c is the conditioning (text embeddings)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embeddings for time"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time, shape (B,)
        Returns:
            embeddings: shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time and context conditioning"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.1,
        num_groups: int = 32,
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )
        
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input, shape (B, C, H, W)
            t_emb: Time embedding, shape (B, time_embed_dim)
        Returns:
            output: shape (B, out_channels, H, W)
        """
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        
        # Add time conditioning
        h = h + self.time_proj(t_emb)[:, :, None, None]
        
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block with optional cross-attention to text"""
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input, shape (B, C, H, W)
        Returns:
            output: shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.proj(out)
        
        return x + out


class CrossAttentionBlock(nn.Module):
    """Cross-attention to text conditioning"""
    
    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 8,
        num_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(num_groups, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input, shape (B, C, H, W)
            context: Text embeddings, shape (B, context_dim)
        Returns:
            output: shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        h = self.norm(x)
        q = self.q(h)
        
        # Get keys and values from context
        kv = self.kv(context)  # (B, 2*C)
        k, v = kv.chunk(2, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # (B, heads, HW, head_dim)
        k = k.view(B, self.num_heads, self.head_dim, 1).transpose(2, 3)  # (B, heads, 1, head_dim)
        v = v.view(B, self.num_heads, self.head_dim, 1).transpose(2, 3)  # (B, heads, 1, head_dim)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)  # (B, heads, HW, 1)
        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        
        # Reshape back
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.proj(out)
        
        return x + out


class DownBlock(nn.Module):
    """Downsampling block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim,
            )
            for i in range(num_layers)
        ])
        
        self.attn_blocks = None
        if use_attention:
            self.attn_blocks = nn.ModuleList([
                AttentionBlock(out_channels)
                for _ in range(num_layers)
            ])
        
        self.cross_attn_blocks = None
        if context_dim is not None:
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(out_channels, context_dim)
                for _ in range(num_layers)
            ])
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            x: Input
            t_emb: Time embedding
            context: Text conditioning
        Returns:
            output: Downsampled output
            skip: Skip connection
        """
        skips = []
        
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, t_emb)
            
            if self.attn_blocks is not None:
                x = self.attn_blocks[i](x)
            
            if self.cross_attn_blocks is not None and context is not None:
                x = self.cross_attn_blocks[i](x, context)
            
            skips.append(x)
        
        x = self.downsample(x)
        
        return x, skips


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels + out_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim,
            )
            for i in range(num_layers)
        ])
        
        self.attn_blocks = None
        if use_attention:
            self.attn_blocks = nn.ModuleList([
                AttentionBlock(out_channels)
                for _ in range(num_layers)
            ])
        
        self.cross_attn_blocks = None
        if context_dim is not None:
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(out_channels, context_dim)
                for _ in range(num_layers)
            ])
    
    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input
            skips: Skip connections from encoder
            t_emb: Time embedding
            context: Text conditioning
        Returns:
            output: Upsampled output
        """
        x = self.upsample(x)
        
        for i, res_block in enumerate(self.res_blocks):
            # Concatenate skip connection only for the first residual layer in this block.
            # ResidualBlock channels are configured as (in_channels + out_channels) for i == 0
            # and out_channels thereafter, so we must not keep concatenating on later layers.
            if i == 0 and len(skips) > 0:
                x = torch.cat([x, skips[-1]], dim=1)
                # Remove the skip we just used so each UpBlock stage consumes one skip set
                skips = skips[:-1]
            
            x = res_block(x, t_emb)
            
            if self.attn_blocks is not None:
                x = self.attn_blocks[i](x)
            
            if self.cross_attn_blocks is not None and context is not None:
                x = self.cross_attn_blocks[i](x, context)
        
        return x


class UNetVelocityPredictor(nn.Module):
    """
    U-Net architecture for predicting velocity field.
    
    This is the main neural network v_θ(x_t, t, c) that learns to match
    the teacher's velocity supervision.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 256,
        channel_mult: tuple = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        context_dim: int = 768,
        time_embed_dim: int = 1024,
    ):
        """
        Args:
            in_channels: Input channels (latent space)
            out_channels: Output channels (velocity field)
            model_channels: Base channel count
            channel_mult: Channel multipliers for each resolution
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions to apply attention
            context_dim: Dimension of text embeddings
            time_embed_dim: Dimension of time embeddings
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    DownBlock(
                        ch,
                        out_ch,
                        time_embed_dim,
                        num_layers=1,
                        use_attention=False,  # Will add attention separately
                        context_dim=context_dim,
                    )
                )
                ch = out_ch
                channels.append(ch)
        
        # Middle
        self.mid_block = nn.ModuleList([
            ResidualBlock(ch, ch, time_embed_dim),
            AttentionBlock(ch),
            CrossAttentionBlock(ch, context_dim),
            ResidualBlock(ch, ch, time_embed_dim),
        ])
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(
                    UpBlock(
                        ch,
                        out_ch,
                        time_embed_dim,
                        num_layers=1,
                        use_attention=False,
                        context_dim=context_dim,
                    )
                )
                ch = out_ch
        
        # Output
        self.output_norm = nn.GroupNorm(32, model_channels)
        self.output_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        
        # Initialize output layer to zero for stability
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity field.
        
        Args:
            x: Current state, shape (B, in_channels, H, W)
            t: Time, shape (B,) or (B, 1, 1, 1)
            context: Text embeddings, shape (B, context_dim)
            
        Returns:
            velocity: Predicted velocity, shape (B, out_channels, H, W)
        """
        # Ensure t is 1D
        if t.dim() > 1:
            t = t.squeeze()
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Input
        h = self.input_conv(x)
        
        # Downsampling with skip connections
        skips = [h]
        for down_block in self.down_blocks:
            h, block_skips = down_block(h, t_emb, context)
            skips.extend(block_skips)
        
        # Middle
        for module in self.mid_block:
            if isinstance(module, (ResidualBlock, AttentionBlock)):
                h = module(h, t_emb) if isinstance(module, ResidualBlock) else module(h)
            elif isinstance(module, CrossAttentionBlock):
                h = module(h, context) if context is not None else h
        
        # Upsampling with skip connections
        for up_block in self.up_blocks:
            h = up_block(h, skips, t_emb, context)
        
        # Output
        h = self.output_norm(h)
        h = torch.nn.functional.silu(h)
        velocity = self.output_conv(h)
        
        return velocity
