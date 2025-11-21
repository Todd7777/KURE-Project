"""
Dataset loaders for Nonlinear Rectified Flows
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
import os
import urllib.request
import zipfile
from tqdm import tqdm
import clip
import torchvision.transforms as transforms

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None


class COCODataset(Dataset):
    """COCO 2017 Captions Dataset"""
    
    def __init__(
        self,
        root: str = "data/coco",
        split: str = "train",
        image_size: int = 256,
        transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        
        # Set paths
        self.image_dir = self.root / "images" / f"{split}2017"
        self.ann_file = self.root / "annotations" / f"captions_{split}2017.json"
        
        # Check if dataset exists
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"COCO image directory not found: {self.image_dir}\n"
                f"Please download the dataset using: python scripts/download_coco.py --root {root} --splits {split}"
            )
        
        if not self.ann_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.ann_file}\n"
                f"Please download the dataset using: python scripts/download_coco.py --root {root} --splits {split}"
            )
        
        # Load annotations
        with open(self.ann_file, "r") as f:
            ann_data = json.load(f)
        
        # Build image_id -> captions mapping
        self.image_info = {img["id"]: img for img in ann_data["images"]}
        self.captions = {}
        for ann in ann_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.captions:
                self.captions[img_id] = []
            self.captions[img_id].append(ann["caption"])
        
        # Create list of (image_id, caption) pairs
        self.samples = []
        for img_id, captions in self.captions.items():
            for caption in captions:
                self.samples.append((img_id, caption))
        
        # Setup transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_id, caption = self.samples[idx]
        img_info = self.image_info[img_id]
        img_path = self.image_dir / img_info["file_name"]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        return {
            "image": image,
            "caption": caption,
            "image_id": img_id,
        }


def create_dataloader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    image_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    root: str = None,
):
    """Create a DataLoader for the specified dataset"""
    
    if dataset_name == "coco":
        if root is None:
            root = "data/coco"
        dataset = COCODataset(root=root, split=split, image_size=image_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # Drop last batch only for training
    )


def download_coco_dataset(root: str = "data/coco", splits: list = ["train", "val"]):
    """Download COCO 2017 dataset"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    # URLs for direct download
    urls = {
        "train": "http://images.cocodataset.org/zips/train2017.zip",
        "val": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }
    
    # Download annotations (needed for all splits)
    if "train" in splits or "val" in splits:
        ann_zip = root / "annotations_trainval2017.zip"
        ann_dir = root / "annotations"
        
        if not ann_dir.exists() or not (ann_dir / "captions_train2017.json").exists():
            print(f"Downloading annotations...")
            _download_file(urls["annotations"], ann_zip)
            print(f"Extracting annotations...")
            _extract_zip(ann_zip, root)
            ann_zip.unlink()  # Remove zip file
    
    # Download images for each split
    for split in splits:
        if split not in ["train", "val"]:
            continue
        
        img_zip = root / f"{split}2017.zip"
        img_dir = root / "images" / f"{split}2017"
        
        if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 0:
            print(f"{split} images already exist, skipping...")
            continue
        
        print(f"Downloading {split} images (~19GB for train, ~1GB for val)...")
        _download_file(urls[split], img_zip)
        print(f"Extracting {split} images...")
        _extract_zip(img_zip, root / "images")
        img_zip.unlink()  # Remove zip file
    
    print("✅ COCO dataset download complete!")


def _download_file(url: str, output_path: Path):
    """Download a file with progress bar"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, (count * block_size * 100) / total_size)
            print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print()  # New line after progress


def _extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file"""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def precompute_text_embeddings(dataset, clip_model, output_path: str = None):
    """Precompute CLIP text embeddings for a dataset (optional optimization)"""
    device = next(clip_model.parameters()).device
    embeddings = []
    
    print("Precomputing text embeddings...")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        caption = sample["caption"]
        
        with torch.no_grad():
            tokens = clip.tokenize([caption], truncate=True).to(device)
            embedding = clip_model.encode_text(tokens)[0]
            embeddings.append(embedding.cpu())
    
    embeddings = torch.stack(embeddings)
    
    if output_path:
        torch.save(embeddings, output_path)
        print(f"Saved embeddings to {output_path}")
    
    return embeddings
