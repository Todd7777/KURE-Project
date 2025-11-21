#!/usr/bin/env python3
"""
Download COCO 2017 dataset.

This script downloads the COCO 2017 images and annotations.
For more information, see: https://cocodataset.org/#download
"""

import sys
sys.path.insert(0, "src")

from pathlib import Path
import argparse
from data.datasets import download_coco_dataset


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017 dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="data/coco",
        help="Root directory to download COCO dataset to",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Which splits to download",
    )
    
    args = parser.parse_args()
    
    print(f"Downloading COCO 2017 dataset to {args.root}")
    print(f"Splits: {args.splits}")
    print("\nThis may take a while depending on your internet connection...")
    print("Train images: ~19GB")
    print("Val images: ~1GB")
    print("Annotations: ~241MB")
    
    try:
        download_coco_dataset(root=args.root, splits=args.splits)
        print("\n✅ COCO dataset download complete!")
        print(f"\nDataset structure:")
        print(f"  {args.root}/")
        print(f"    images/")
        print(f"      train2017/")
        print(f"      val2017/")
        print(f"    annotations/")
        print(f"      captions_train2017.json")
        print(f"      captions_val2017.json")
    except Exception as e:
        print(f"\n❌ Error downloading COCO dataset: {e}")
        print("\nPlease download manually from: https://cocodataset.org/#download")
        print("And extract to the following structure:")
        print(f"  {args.root}/images/train2017/")
        print(f"  {args.root}/images/val2017/")
        print(f"  {args.root}/annotations/captions_train2017.json")
        print(f"  {args.root}/annotations/captions_val2017.json")
        sys.exit(1)


if __name__ == "__main__":
    main()














