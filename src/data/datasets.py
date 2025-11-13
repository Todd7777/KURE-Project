"""
Dataset loaders for Nonlinear Rectified Flows.

Supports COCO, CC3M, and LAION datasets with text-image pairs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import torchvision.transforms as transforms

try:
    from pycocotools.coco import COCO
    PYCOCO_AVAILABLE = True
except ImportError:
    PYCOCO_AVAILABLE = False
    print("Warning: pycocotools not available. COCO dataset will not work.")


class COCODataset(Dataset):
    """
    COCO dataset loader for text-image pairs.
    
    Args:
        root: Root directory containing COCO data
        split: 'train' or 'val'
        image_size: Target image size
        transform: Optional transform to apply to images
    """
    
    def __init__(
        self,
        root: str = "data/coco",
        split: str = "train",
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        
        # Set up paths
        if split == "train":
            self.image_dir = self.root / "images" / "train2017"
            self.annot_file = self.root / "annotations" / "captions_train2017.json"
        else:
            self.image_dir = self.root / "images" / "val2017"
            self.annot_file = self.root / "annotations" / "captions_val2017.json"
        
        # Load annotations
        if not self.annot_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.annot_file}\n"
                f"Please download COCO dataset to {self.root}"
            )
        
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"COCO image directory not found: {self.image_dir}\n"
                f"Please download COCO dataset to {self.root}"
            )
        
        # Load COCO annotations
        if PYCOCO_AVAILABLE:
            self.coco = COCO(str(self.annot_file))
            self.image_ids = list(self.coco.imgs.keys())
            # Get all annotations (images can have multiple captions)
            self.annotations = []
            for img_id in self.image_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                for ann in anns:
                    self.annotations.append({
                        'image_id': img_id,
                        'caption': ann['caption'],
                        'file_name': self.coco.imgs[img_id]['file_name']
                    })
        else:
            # Fallback: load from JSON directly
            with open(self.annot_file, 'r') as f:
                data = json.load(f)
            self.annotations = []
            images_dict = {img['id']: img for img in data['images']}
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id in images_dict:
                    self.annotations.append({
                        'image_id': img_id,
                        'caption': ann['caption'],
                        'file_name': images_dict[img_id]['file_name']
                    })
        
        # Set up transforms
        if transform is None:
            from .augmentation import get_transforms
            self.transform = get_transforms(
                image_size=image_size,
                is_train=(split == "train")
            )
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.annotations)} {split} samples from COCO")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        ann = self.annotations[idx]
        
        # Load image
        image_path = self.image_dir / ann['file_name']
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return dictionary with image and caption
        return {
            'image': image,
            'caption': ann['caption'],
            'image_id': ann['image_id'],
        }


def create_dataloader(
    dataset_name: str = "coco",
    split: str = "train",
    batch_size: int = 64,
    image_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = True,
    root: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.
    
    Args:
        dataset_name: Name of dataset ('coco', 'cc3m', 'laion')
        split: 'train' or 'val'
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
        root: Root directory for dataset (defaults to data/{dataset_name})
        
    Returns:
        DataLoader instance
    """
    if root is None:
        root = f"data/{dataset_name}"
    
    if dataset_name == "coco":
        dataset = COCODataset(
            root=root,
            split=split,
            image_size=image_size,
        )
    elif dataset_name == "cc3m":
        # TODO: Implement CC3M dataset
        raise NotImplementedError("CC3M dataset not yet implemented")
    elif dataset_name == "laion":
        # TODO: Implement LAION dataset
        raise NotImplementedError("LAION dataset not yet implemented")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    
    return dataloader


def precompute_text_embeddings(
    dataset: Dataset,
    clip_model,
    output_path: str,
    batch_size: int = 256,
):
    """
    Precompute CLIP text embeddings for a dataset.
    
    Args:
        dataset: Dataset instance
        clip_model: CLIP model
        output_path: Path to save embeddings
        batch_size: Batch size for processing
    """
    import torch
    import clip
    from tqdm import tqdm
    
    device = next(clip_model.parameters()).device
    embeddings = []
    captions = []
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            if isinstance(batch, dict):
                batch_captions = batch['caption']
            else:
                batch_captions = batch
            
            # Tokenize and encode
            if isinstance(batch_captions, list):
                try:
                    import clip
                    tokens = clip.tokenize(batch_captions, truncate=True).to(device)
                    batch_embeddings = clip_model.encode_text(tokens)
                    embeddings.append(batch_embeddings.cpu())
                    captions.extend(batch_captions)
                except Exception as e:
                    print(f"Warning: Error encoding captions: {e}")
                    continue
    
    # Save embeddings
    if embeddings:
        torch.save({
            'embeddings': torch.cat(embeddings, dim=0),
            'captions': captions,
        }, output_path)
        print(f"Saved {len(captions)} embeddings to {output_path}")
    else:
        print("Warning: No embeddings computed")


def download_coco_dataset(
    root: str = "data/coco",
    splits: List[str] = ["train", "val"],
):
    """
    Download COCO 2017 dataset.
    
    Args:
        root: Root directory to download to
        splits: Which splits to download ('train', 'val')
    """
    import urllib.request
    import zipfile
    from tqdm import tqdm
    import shutil
    
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # URLs for COCO dataset
    urls = {
        'train': 'http://images.cocodataset.org/zips/train2017.zip',
        'val': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    }
    
    # Download annotations
    print("Downloading annotations...")
    annot_zip = root_path / "annotations_trainval2017.zip"
    if not annot_zip.exists():
        urllib.request.urlretrieve(urls['annotations'], annot_zip)
    
    # Extract annotations
    print("Extracting annotations...")
    with zipfile.ZipFile(annot_zip, 'r') as zip_ref:
        zip_ref.extractall(root_path)
    
    # Download and extract images for each split
    for split in splits:
        if split not in ['train', 'val']:
            continue
        
        print(f"Downloading {split} images...")
        split_zip = root_path / f"{split}2017.zip"
        if not split_zip.exists():
            urllib.request.urlretrieve(urls[split], split_zip)
        
        print(f"Extracting {split} images...")
        images_dir = root_path / "images"
        images_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(split_zip, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
    
    print("COCO dataset download complete!")
