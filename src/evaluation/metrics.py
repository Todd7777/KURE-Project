"""
Evaluation Metrics for NRF

This module implements:
- Fréchet Inception Distance (FID)
- Inception Score (IS)
- CLIPScore for text-image alignment
- Compositional evaluation metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import linalg
from torchvision.models import inception_v3
import clip
from tqdm import tqdm


class InceptionFeatureExtractor(nn.Module):
    """Extract features from Inception-v3 for FID computation"""
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        self.inception.to(device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract Inception features.
        
        Args:
            x: Images, shape (B, 3, 299, 299)
            
        Returns:
            features: Shape (B, 2048)
        """
        return self.inception(x)
    
    @torch.no_grad()
    def extract_features_from_images(
        self,
        images: torch.Tensor,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Args:
            images: Images, shape (N, 3, H, W), values in [-1, 1]
            batch_size: Batch size for processing
            
        Returns:
            features: Shape (N, 2048)
        """
        # Resize to 299x299 for Inception
        images = torch.nn.functional.interpolate(
            images,
            size=(299, 299),
            mode="bilinear",
            align_corners=False,
        )
        
        features_list = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            feats = self.forward(batch)
            features_list.append(feats.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)


def calculate_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Calculate Fréchet Inception Distance.
    
    FID = ||mu_real - mu_fake||^2 + Tr(Sigma_real + Sigma_fake - 2*sqrt(Sigma_real @ Sigma_fake))
    
    Args:
        real_features: Real image features, shape (N, D)
        fake_features: Generated image features, shape (M, D)
        eps: Small constant for numerical stability
        
    Returns:
        fid: FID score (lower is better)
    """
    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute mean difference
    diff = mu_real - mu_fake
    
    # Compute matrix sqrt using SVD
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Compute FID
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return float(fid)


def calculate_inception_score(
    images: torch.Tensor,
    batch_size: int = 32,
    splits: int = 10,
    device: str = "cuda",
) -> Tuple[float, float]:
    """
    Calculate Inception Score.
    
    IS = exp(E[KL(p(y|x) || p(y))])
    
    Args:
        images: Generated images, shape (N, 3, H, W)
        batch_size: Batch size for processing
        splits: Number of splits for computing mean and std
        device: Device to use
        
    Returns:
        mean_is: Mean Inception Score
        std_is: Standard deviation of IS
    """
    # Load Inception model
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.eval()
    inception.to(device)
    
    # Resize images
    images = torch.nn.functional.interpolate(
        images,
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
    )
    
    # Get predictions
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            pred = torch.nn.functional.softmax(inception(batch), dim=1)
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Compute IS for each split
    split_scores = []
    
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
        
        split_scores.append(np.exp(np.mean(scores)))
    
    return float(np.mean(split_scores)), float(np.std(split_scores))


class CLIPScoreEvaluator:
    """Evaluate text-image alignment using CLIP"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Args:
            model_name: CLIP model name
            device: Device to use
        """
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
    
    @torch.no_grad()
    def compute_clip_score(
        self,
        images: torch.Tensor,
        texts: List[str],
        batch_size: int = 32,
    ) -> float:
        """
        Compute CLIPScore: average cosine similarity between image and text embeddings.
        
        Args:
            images: Images, shape (N, 3, H, W), values in [-1, 1]
            texts: List of N text prompts
            batch_size: Batch size for processing
            
        Returns:
            clip_score: Average CLIPScore (higher is better)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        
        # Normalize images to [0, 1]
        images = (images + 1) / 2
        
        # Resize to CLIP input size (224x224)
        images = torch.nn.functional.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        
        scores = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size].to(self.device)
            batch_texts = texts[i:i+batch_size]
            
            # Encode images
            image_features = self.model.encode_image(batch_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Encode texts
            text_tokens = clip.tokenize(batch_texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (image_features * text_features).sum(dim=-1)
            scores.append(similarity.cpu().numpy())
        
        scores = np.concatenate(scores)
        return float(np.mean(scores))
    
    @torch.no_grad()
    def compute_clip_score_per_category(
        self,
        images: torch.Tensor,
        texts: List[str],
        categories: List[str],
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Compute CLIPScore broken down by category.
        
        Args:
            images: Images
            texts: Text prompts
            categories: Category labels for each prompt
            batch_size: Batch size
            
        Returns:
            scores: Dictionary mapping category to CLIPScore
        """
        assert len(images) == len(texts) == len(categories)
        
        # Normalize images
        images = (images + 1) / 2
        images = torch.nn.functional.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        
        # Group by category
        category_scores = {}
        unique_categories = set(categories)
        
        for category in unique_categories:
            indices = [i for i, c in enumerate(categories) if c == category]
            cat_images = images[indices]
            cat_texts = [texts[i] for i in indices]
            
            scores = []
            
            for i in range(0, len(cat_images), batch_size):
                batch_images = cat_images[i:i+batch_size].to(self.device)
                batch_texts = cat_texts[i:i+batch_size]
                
                image_features = self.model.encode_image(batch_images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                text_tokens = clip.tokenize(batch_texts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features * text_features).sum(dim=-1)
                scores.append(similarity.cpu().numpy())
            
            category_scores[category] = float(np.mean(np.concatenate(scores)))
        
        return category_scores


class NRFEvaluator:
    """
    Comprehensive evaluator for NRF models.
    
    Computes FID, IS, and CLIPScore for generated images.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: Device to use
        """
        self.device = device
        self.inception_extractor = InceptionFeatureExtractor(device)
        self.clip_evaluator = CLIPScoreEvaluator(device=device)
    
    def evaluate(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        prompts: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation.
        
        Args:
            real_images: Real images, shape (N, 3, H, W)
            fake_images: Generated images, shape (M, 3, H, W)
            prompts: Text prompts for generated images
            categories: Category labels for compositional evaluation
            batch_size: Batch size for processing
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        metrics = {}
        
        print("Computing FID...")
        # Extract Inception features
        real_features = self.inception_extractor.extract_features_from_images(
            real_images, batch_size
        )
        fake_features = self.inception_extractor.extract_features_from_images(
            fake_images, batch_size
        )
        
        # Compute FID
        fid = calculate_fid(real_features, fake_features)
        metrics["fid"] = fid
        
        print("Computing Inception Score...")
        # Compute IS
        is_mean, is_std = calculate_inception_score(
            fake_images, batch_size, device=self.device
        )
        metrics["is_mean"] = is_mean
        metrics["is_std"] = is_std
        
        # Compute CLIPScore if prompts provided
        if prompts is not None:
            print("Computing CLIPScore...")
            clip_score = self.clip_evaluator.compute_clip_score(
                fake_images, prompts, batch_size
            )
            metrics["clip_score"] = clip_score
            
            # Per-category CLIPScore
            if categories is not None:
                print("Computing per-category CLIPScore...")
                category_scores = self.clip_evaluator.compute_clip_score_per_category(
                    fake_images, prompts, categories, batch_size
                )
                for cat, score in category_scores.items():
                    metrics[f"clip_score_{cat}"] = score
        
        return metrics
    
    def evaluate_at_steps(
        self,
        model,
        vae,
        prompts: List[str],
        text_embeddings: torch.Tensor,
        real_images: torch.Tensor,
        categories: Optional[List[str]] = None,
        step_budgets: List[int] = [1, 2, 4, 8],
        num_samples_per_prompt: int = 1,
        batch_size: int = 8,
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model at different step budgets.
        
        Args:
            model: NRF model
            vae: VAE for decoding
            prompts: Text prompts
            text_embeddings: Precomputed text embeddings
            real_images: Real images for FID
            categories: Category labels
            step_budgets: List of step counts to evaluate
            num_samples_per_prompt: Number of samples per prompt
            batch_size: Batch size for generation
            
        Returns:
            results: Dictionary mapping step count to metrics
        """
        results = {}
        
        for num_steps in step_budgets:
            print(f"\nEvaluating with {num_steps} steps...")
            
            # Generate images
            all_fake_images = []
            all_prompts = []
            all_categories = []
            
            for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
                batch_prompts = prompts[i:i+batch_size]
                batch_embeddings = text_embeddings[i:i+batch_size]
                batch_categories = categories[i:i+batch_size] if categories else None
                
                for _ in range(num_samples_per_prompt):
                    # Sample from model
                    with torch.no_grad():
                        latents = model.sample(
                            batch_size=len(batch_prompts),
                            shape=(vae.decoder.latent_channels, 64, 64),
                            context=batch_embeddings,
                            num_steps=num_steps,
                            solver="euler",
                            use_ema=True,
                        )
                        
                        # Decode to images
                        fake_images = vae.decode(latents)
                    
                    all_fake_images.append(fake_images.cpu())
                    all_prompts.extend(batch_prompts)
                    if batch_categories:
                        all_categories.extend(batch_categories)
            
            # Concatenate all generated images
            all_fake_images = torch.cat(all_fake_images, dim=0)
            
            # Evaluate
            metrics = self.evaluate(
                real_images,
                all_fake_images,
                prompts=all_prompts,
                categories=all_categories if all_categories else None,
                batch_size=batch_size,
            )
            
            results[num_steps] = metrics
            
            print(f"Results for {num_steps} steps:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
        return results
