"""
Dataset Download Helper
========================
Downloads and organizes the two primary datasets for this project:
    1. Agroscapes (via Agronav repo - already a submodule)
    2. GD-YOLOv10n-seg dataset (contact authors - instructions below)

Usage:
    conda activate torch_sm120
    python download_datasets.py --dataset vegann --output ../../data/raw/

Supported datasets:
    vegann    - VegAnn vegetation segmentation dataset (Zenodo, CC-BY)
    junfeng   - JunfengGaolab CropRowDetection dataset (GitHub)
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def download_vegann(output_dir: Path):
    """Download VegAnn dataset from Zenodo."""
    print("Downloading VegAnn dataset from Zenodo...")
    url = "https://zenodo.org/records/7636408/files/VegAnn.zip"
    dest = output_dir / "vegann"
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / "VegAnn.zip"
    subprocess.run(["wget", "-c", url, "-O", str(zip_path)], check=True)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)

    os.remove(zip_path)
    print(f"✅ VegAnn saved to {dest}")


def download_junfeng(output_dir: Path):
    """Clone JunfengGaolab CropRowDetection dataset."""
    print("Cloning JunfengGaolab CropRowDetection dataset...")
    dest = output_dir / "junfeng_croprow"

    if dest.exists():
        print(f"⚠️  {dest} already exists, skipping clone")
    else:
        subprocess.run([
            "git", "clone",
            "https://github.com/JunfengGaolab/CropRowDetection.git",
            str(dest)
        ], check=True)

    print(f"✅ JunfengGaolab dataset saved to {dest}")


def download_sam2_checkpoints(output_dir: Path):
    """Download SAM2 model checkpoints."""
    print("Downloading SAM2 checkpoints...")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "sam2.1_hiera_large.pt":
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sam2.1_hiera_base_plus.pt":
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    }

    for name, url in checkpoints.items():
        dest = output_dir / name
        if dest.exists():
            print(f"  ⚠️  {name} already exists, skipping")
            continue
        print(f"  Downloading {name}...")
        subprocess.run(["wget", "-c", url, "-O", str(dest)], check=True)
        print(f"  ✅ {name} saved")


def main():
    parser = argparse.ArgumentParser(description="Dataset download helper")
    parser.add_argument(
        "--dataset",
        choices=["vegann", "junfeng", "sam2_checkpoints", "all"],
        required=True,
        help="Which dataset to download"
    )
    parser.add_argument(
        "--output",
        default="../../data/raw",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--checkpoints",
        default="../../../checkpoints/sam2",
        help="Output directory for SAM2 checkpoints"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("vegann", "all"):
        download_vegann(output_dir)

    if args.dataset in ("junfeng", "all"):
        download_junfeng(output_dir)

    if args.dataset in ("sam2_checkpoints", "all"):
        download_sam2_checkpoints(Path(args.checkpoints))

    print("\n✅ All requested downloads complete!")
    print("\n📋 GD-YOLOv10n-seg dataset:")
    print("   This dataset requires contacting the authors directly.")
    print("   Paper: https://www.mdpi.com/2077-0472/15/7/796")
    print("   Email the corresponding author to request access.")


if __name__ == "__main__":
    main()
