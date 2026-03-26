"""
SAM2 Pseudo-Labeling Pipeline
==============================
Takes raw field images and generates YOLO segmentation format labels
using SAM2's automatic mask generator.

Usage:
    conda activate torch_sm120
    python sam_pipeline.py \
        --input  ../../data/raw/agroscapes \
        --output ../../data/labels \
        --checkpoint ../../../checkpoints/sam2/sam2.1_hiera_large.pt \
        --vis    ../../data/visualizations

Output format:
    YOLO segmentation .txt files (one per image)
    Each line: <class_id> <x1> <y1> <x2> <y2> ... (normalized polygon)
"""

import argparse
import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ── Config ─────────────────────────────────────────────────────────────────────
CROP_CLASS_ID  = 0  # YOLO class id for crop/vegetation
SOIL_CLASS_ID  = 1  # YOLO class id for soil (optional)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SAM2_CONFIG_MAP = {
    "sam2.1_hiera_large.pt":     "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_small.pt":     "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_tiny.pt":      "configs/sam2.1/sam2.1_hiera_t.yaml",
}


def build_generator(checkpoint_path: str, device: str) -> SAM2AutomaticMaskGenerator:
    ckpt_name = Path(checkpoint_path).name
    model_cfg = SAM2_CONFIG_MAP.get(ckpt_name)
    if model_cfg is None:
        raise ValueError(f"Unknown checkpoint: {ckpt_name}. "
                         f"Valid options: {list(SAM2_CONFIG_MAP.keys())}")

    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)

    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=500,
    )


def mask_to_yolo_polygon(mask: np.ndarray, img_w: int, img_h: int) -> list:
    """Convert binary mask to YOLO normalized polygon format."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    # Use largest contour
    contour = max(contours, key=cv2.contourArea)
    # Simplify polygon
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx  = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        return []

    # Normalize to [0, 1]
    points = approx.reshape(-1, 2)
    normalized = []
    for x, y in points:
        normalized.extend([x / img_w, y / img_h])
    return normalized


def is_vegetation_mask(mask: np.ndarray, bgr_image: np.ndarray,
                        exg_thresh: float = 0.05) -> bool:
    """
    Filter: keep masks that overlap significantly with ExG vegetation signal.
    Rejects sky, soil-only, and non-crop masks.
    """
    b, g, r = cv2.split(bgr_image.astype(np.float32))
    total = r + g + b + 1e-6
    exg = (2.0 * g - r - b) / total

    masked_exg = exg[mask > 0]
    if len(masked_exg) == 0:
        return False

    mean_exg = float(masked_exg.mean())
    return mean_exg > exg_thresh


def process_image(
    img_path: Path,
    generator: SAM2AutomaticMaskGenerator,
    output_dir: Path,
    vis_dir: Path = None,
) -> int:
    """Process one image. Returns number of masks written."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"  ⚠️  Could not read {img_path.name}, skipping")
        return 0

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]

    masks = generator.generate(rgb)

    yolo_lines = []
    vis_mask   = np.zeros((h, w, 3), dtype=np.uint8)

    for mask_data in masks:
        mask = mask_data["segmentation"]

        # Filter to vegetation masks only
        if not is_vegetation_mask(mask, bgr):
            continue

        polygon = mask_to_yolo_polygon(mask, w, h)
        if len(polygon) < 6:
            continue

        coords_str = " ".join(f"{v:.6f}" for v in polygon)
        yolo_lines.append(f"{CROP_CLASS_ID} {coords_str}")

        # Visualization
        if vis_dir:
            color = np.random.randint(80, 255, 3).tolist()
            vis_mask[mask] = color

    # Write YOLO label file
    label_path = output_dir / (img_path.stem + ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # Write visualization
    if vis_dir and len(yolo_lines) > 0:
        overlay = cv2.addWeighted(bgr, 0.6, vis_mask, 0.4, 0)
        vis_path = vis_dir / (img_path.stem + "_sam.jpg")
        cv2.imwrite(str(vis_path), overlay)

    return len(yolo_lines)


def main():
    parser = argparse.ArgumentParser(description="SAM2 pseudo-labeling pipeline")
    parser.add_argument("--input",      required=True, help="Input image directory")
    parser.add_argument("--output",     required=True, help="Output YOLO labels directory")
    parser.add_argument("--checkpoint", required=True, help="SAM2 checkpoint .pt file")
    parser.add_argument("--vis",        default=None,  help="Visualization output directory")
    parser.add_argument("--device",     default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    vis_dir    = Path(args.vis) if args.vis else None

    output_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    images = [
        p for p in sorted(input_dir.rglob("*"))
        if p.suffix.lower() in IMG_EXTENSIONS
    ]
    if not images:
        print(f"❌ No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images")
    print(f"Loading SAM2 from {args.checkpoint}...")

    device    = args.device if torch.cuda.is_available() else "cpu"
    generator = build_generator(args.checkpoint, device)

    print(f"Running on: {device}")
    print(f"Output labels → {output_dir}")
    if vis_dir:
        print(f"Visualizations → {vis_dir}")
    print()

    total_masks = 0
    for img_path in tqdm(images, desc="Labeling"):
        n = process_image(img_path, generator, output_dir, vis_dir)
        total_masks += n

    print(f"\n✅ Done! {total_masks} total masks written for {len(images)} images")
    print(f"   Labels saved to: {output_dir}")


if __name__ == "__main__":
    main()
