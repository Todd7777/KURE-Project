"""
Sampling script for generating images with NRF

Usage:
    python scripts/sample.py \
        --checkpoint checkpoints/nrf_spline_best.pt \
        --prompt "A red cube on top of a blue sphere" \
        --steps 4 \
        --num_samples 16
"""

import sys
sys.path.insert(0, "src")

import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
import clip

from models.nrf_base import NonlinearRectifiedFlow
from models.vae import create_vae


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """Load NRF model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # This is simplified - in practice, you'd need to recreate the full architecture
    # from the config stored in the checkpoint
    config = checkpoint.get("config", {})
    
    # For now, return checkpoint and config
    return checkpoint, config


def encode_prompts(prompts: list, device: str):
    """Encode text prompts using CLIP"""
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    with torch.no_grad():
        tokens = clip.tokenize(prompts, truncate=True).to(device)
        embeddings = clip_model.encode_text(tokens)
    
    return embeddings


def generate_images(
    model,
    vae,
    prompts: list,
    num_samples: int = 4,
    num_steps: int = 4,
    solver: str = "euler",
    guidance_scale: float = 7.5,
    device: str = "cuda",
):
    """
    Generate images from text prompts.
    
    Args:
        model: NRF model
        vae: VAE for decoding
        prompts: List of text prompts
        num_samples: Number of samples per prompt
        num_steps: Number of sampling steps
        solver: ODE solver
        guidance_scale: Classifier-free guidance scale
        device: Device to use
        
    Returns:
        images: Generated images, shape (len(prompts)*num_samples, 3, H, W)
    """
    # Encode prompts
    text_embeddings = encode_prompts(prompts, device)
    
    all_images = []
    
    for i, (prompt, embedding) in enumerate(zip(prompts, text_embeddings)):
        print(f"Generating {num_samples} samples for: '{prompt}'")
        
        # Repeat embedding for multiple samples
        batch_embeddings = embedding.unsqueeze(0).repeat(num_samples, 1)
        
        with torch.no_grad():
            # Sample from model
            latents = model.sample(
                batch_size=num_samples,
                shape=(vae.decoder.latent_channels, 64, 64),
                context=batch_embeddings,
                num_steps=num_steps,
                solver=solver,
                use_ema=True,
                guidance_scale=guidance_scale,
            )
            
            # Decode to images
            images = vae.decode(latents)
        
        all_images.append(images.cpu())
    
    return torch.cat(all_images, dim=0)


def save_images(images: torch.Tensor, output_dir: Path, prompts: list, num_samples: int):
    """Save generated images"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Save individual images
    idx = 0
    for i, prompt in enumerate(prompts):
        prompt_dir = output_dir / f"prompt_{i:03d}"
        prompt_dir.mkdir(exist_ok=True)
        
        # Save prompt
        with open(prompt_dir / "prompt.txt", "w") as f:
            f.write(prompt)
        
        # Save samples
        for j in range(num_samples):
            img = images[idx]
            save_image(img, prompt_dir / f"sample_{j:03d}.png")
            idx += 1
    
    # Save grid
    grid = make_grid(images, nrow=num_samples, padding=2, normalize=False)
    save_image(grid, output_dir / "grid.png")
    
    print(f"Saved {len(images)} images to {output_dir}")


def main(args):
    """Main sampling function"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Create VAE
    print("Creating VAE...")
    vae = create_vae(
        image_size=args.image_size,
        latent_size=64,
        in_channels=3,
        latent_channels=4,
    ).to(device)
    
    if config.get("vae_checkpoint"):
        vae.load_state_dict(torch.load(config["vae_checkpoint"], map_location=device))
    
    vae.eval()
    
    # Load NRF model
    # Note: This is simplified - you'd need to recreate the full architecture
    model = checkpoint["model"]  # Placeholder
    
    # Get prompts
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        # Default prompts
        prompts = [
            "A red cube on top of a blue sphere",
            "A small wooden chair next to a large metal table",
            "A cat without a collar",
            "A green apple left of a yellow banana",
        ]
    
    print(f"Generating images for {len(prompts)} prompts...")
    
    # Generate images
    images = generate_images(
        model=model,
        vae=vae,
        prompts=prompts,
        num_samples=args.num_samples,
        num_steps=args.steps,
        solver=args.solver,
        guidance_scale=args.guidance_scale,
        device=device,
    )
    
    # Save images
    output_dir = Path(args.output_dir)
    save_images(images, output_dir, prompts, args.num_samples)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with NRF")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Single text prompt")
    parser.add_argument("--prompts_file", type=str, default=None, help="File with prompts (one per line)")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples per prompt")
    parser.add_argument("--steps", type=int, default=4, help="Number of sampling steps")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun", "rk4"])
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)
