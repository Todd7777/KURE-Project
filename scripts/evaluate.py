"""
Evaluation script for Nonlinear Rectified Flows

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset coco --steps 1 2 4 8
"""

import sys
sys.path.insert(0, "src")

import torch
import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm

from models.nrf_base import NonlinearRectifiedFlow
from models.vae import create_vae
from evaluation.metrics import NRFEvaluator
from evaluation.compositional_suite import CompositionalPromptGenerator, CompositionalEvaluator
from data.datasets import create_dataloader
import clip


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    # Recreate model architecture (simplified - you'd need to import the actual creation logic)
    print("Loading model architecture...")
    # This would use the same logic as train.py to recreate the model
    # For brevity, assuming model is saved with full architecture
    
    return checkpoint, config


def evaluate_standard_metrics(
    args,
    model,
    vae,
    device: str,
):
    """Evaluate standard metrics (FID, IS, CLIPScore)"""
    
    print("Loading evaluation data...")
    # Load real images
    real_loader = create_dataloader(
        dataset_name=args.dataset,
        split="val",
        batch_size=32,
        image_size=256,
        num_workers=4,
        shuffle=False,
    )
    
    # Collect real images
    real_images = []
    prompts = []
    
    for batch in tqdm(real_loader, desc="Loading real images"):
        real_images.append(batch["image"])
        if "caption" in batch:
            prompts.extend(batch["caption"])
        
        if len(real_images) * 32 >= args.num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:args.num_samples]
    
    # Create evaluator
    evaluator = NRFEvaluator(device=device)
    
    # Evaluate at different step budgets
    step_budgets = [int(s) for s in args.steps.split()]
    
    results = {}
    
    for num_steps in step_budgets:
        print(f"\n{'='*50}")
        print(f"Evaluating with {num_steps} steps")
        print(f"{'='*50}")
        
        # Generate images
        fake_images = []
        
        for i in tqdm(range(0, args.num_samples, args.batch_size), desc="Generating"):
            batch_size = min(args.batch_size, args.num_samples - i)
            
            # Get text embeddings if available
            batch_prompts = prompts[i:i+batch_size] if prompts else None
            
            # TODO: Encode prompts to embeddings
            context = None  # Would need CLIP text encoder
            
            with torch.no_grad():
                # Sample from model
                latents = model.sample(
                    batch_size=batch_size,
                    shape=(vae.decoder.latent_channels, 64, 64),
                    context=context,
                    num_steps=num_steps,
                    solver=args.solver,
                    use_ema=True,
                )
                
                # Decode to images
                images = vae.decode(latents)
            
            fake_images.append(images.cpu())
        
        fake_images = torch.cat(fake_images, dim=0)
        
        # Evaluate
        metrics = evaluator.evaluate(
            real_images=real_images,
            fake_images=fake_images,
            prompts=prompts[:args.num_samples] if prompts else None,
            batch_size=32,
        )
        
        results[num_steps] = metrics
        
        print(f"\nResults for {num_steps} steps:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return results


def evaluate_compositional(
    args,
    model,
    vae,
    device: str,
):
    """Evaluate on compositional suite"""
    
    print("\n" + "="*50)
    print("Compositional Evaluation")
    print("="*50)
    
    # Generate or load compositional prompts
    generator = CompositionalPromptGenerator()
    
    if args.compositional_suite:
        suite = generator.load_suite(args.compositional_suite)
    else:
        suite = generator.generate_full_suite()
    
    # Load CLIP for verification
    clip_model, _ = clip.load("ViT-B/32", device=device)
    evaluator = CompositionalEvaluator(clip_model, device=device)
    
    results = {}
    
    for category, prompts in suite.items():
        print(f"\nEvaluating {category} ({len(prompts)} prompts)...")
        
        # Generate images
        images = []
        
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Generating {category}"):
            batch_prompts = prompts[i:i+args.batch_size]
            batch_size = len(batch_prompts)
            
            # TODO: Encode prompts to embeddings
            context = None
            
            with torch.no_grad():
                latents = model.sample(
                    batch_size=batch_size,
                    shape=(vae.decoder.latent_channels, 64, 64),
                    context=context,
                    num_steps=args.compositional_steps,
                    solver=args.solver,
                    use_ema=True,
                )
                
                batch_images = vae.decode(latents)
            
            images.append(batch_images.cpu())
        
        images = torch.cat(images, dim=0)
        
        # Evaluate
        category_results = evaluator.evaluate_suite(images, prompts)
        results[category] = category_results
        
        print(f"Results for {category}:")
        for key, value in category_results.items():
            print(f"  {key}: {value:.4f}")
    
    return results


def main(args):
    """Main evaluation function"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint, config = load_model(args.checkpoint, device)
    
    # Create VAE
    print("Creating VAE...")
    vae = create_vae(
        image_size=256,
        latent_size=64,
        in_channels=3,
        latent_channels=4,
    ).to(device)
    
    if config.get("vae_checkpoint"):
        vae.load_state_dict(torch.load(config["vae_checkpoint"], map_location=device))
    
    vae.eval()
    
    # Load NRF model
    # This is simplified - you'd need to recreate the full model architecture
    model = checkpoint["model"]  # Placeholder
    
    # Run evaluations
    results = {}
    
    if args.eval_type in ["all", "standard"]:
        standard_results = evaluate_standard_metrics(args, model, vae, device)
        results["standard"] = standard_results
    
    if args.eval_type in ["all", "compositional"]:
        compositional_results = evaluate_compositional(args, model, vae, device)
        results["compositional"] = compositional_results
    
    # Save results
    output_path = Path(args.output_dir) / f"results_{args.dataset}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if "standard" in results:
        print("\nStandard Metrics:")
        for steps, metrics in results["standard"].items():
            print(f"\n  {steps} steps:")
            print(f"    FID: {metrics.get('fid', 'N/A'):.2f}")
            print(f"    IS: {metrics.get('is_mean', 'N/A'):.2f}")
            print(f"    CLIPScore: {metrics.get('clip_score', 'N/A'):.4f}")
    
    if "compositional" in results:
        print("\nCompositional Metrics:")
        for category, metrics in results["compositional"].items():
            print(f"\n  {category}:")
            for key, value in metrics.items():
                print(f"    {key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Nonlinear Rectified Flow")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset to evaluate on")
    parser.add_argument("--eval_type", type=str, default="all", choices=["all", "standard", "compositional"])
    parser.add_argument("--steps", type=str, default="1 2 4 8", help="Step budgets to evaluate")
    parser.add_argument("--compositional_steps", type=int, default=4, help="Steps for compositional eval")
    parser.add_argument("--compositional_suite", type=str, default=None, help="Path to compositional suite JSON")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples for FID/IS")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun", "rk4"])
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    main(args)
