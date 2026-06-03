"""
Preprocess training images for TensorRT INT8 calibration.

Outputs float32 CHW numpy arrays to training/calibration/images/.
Preprocessing MUST match the runtime GStreamer/DeepStream pipeline:
  - Resize to imgsz x imgsz (bilinear)
  - BGR -> RGB
  - Normalize: (pixel/255 - mean) / std  with ImageNet mean/std
  - Layout: CHW, dtype float32

Usage (run from training/ directory):
  python scripts/04_prepare_calibration.py --n 50 --imgsz 480
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr, imgsz):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/raw/images")
    parser.add_argument("--split", default="dataset/coco/train.json",
                        help="COCO train.json — limits candidates to train split only")
    parser.add_argument("--out", default="calibration")
    parser.add_argument("--n", type=int, default=50, help="Number of calibration images")
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_json = Path(args.split)
    if split_json.exists():
        with open(split_json) as f:
            coco = json.load(f)
        train_filenames = {img["file_name"] for img in coco["images"]}
        candidates = [
            data_dir / fn
            for fn in sorted(train_filenames)
            if (data_dir / fn).exists()
        ]
        print(f"Using {len(candidates)} images from train split")
    else:
        print(f"WARNING: {split_json} not found; using all images in {data_dir}")
        candidates = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))

    if not candidates:
        print(f"ERROR: No images found in {data_dir}")
        sys.exit(1)

    random.seed(args.seed)
    selected = random.sample(candidates, min(args.n, len(candidates)))
    print(f"Preprocessing {len(selected)} images -> {out_dir}")

    saved_paths = []
    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: could not read {img_path}, skipping")
            continue
        tensor = preprocess(img, args.imgsz)
        out_file = out_dir / (img_path.stem + ".npy")
        np.save(str(out_file), tensor)
        saved_paths.append(str(out_file.resolve()))

    list_file = Path(args.out) / "calib_list.txt"
    list_file.write_text("\n".join(saved_paths) + "\n")

    print(f"Saved {len(saved_paths)} calibration tensors")
    print(f"Shape per tensor: (3, {args.imgsz}, {args.imgsz}), float32")
    print(f"List file: {list_file}")
    print()
    print("IMPORTANT: Preprocessing uses ImageNet mean/std normalization.")
    print("Verify this matches the Heimdall GStreamer pipeline before building INT8 engine.")


if __name__ == "__main__":
    main()
