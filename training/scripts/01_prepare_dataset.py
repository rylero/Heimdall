"""
Extract Label Studio YOLO export zip and convert to COCO JSON format.

Usage (run from training/ directory):
  python scripts/01_prepare_dataset.py \
    --zip "C:/Users/ryan/Downloads/project-5-at-2026-01-17-20-11-0f80986e.zip"
"""
import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import cv2


def yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h):
    """YOLO normalized (cx,cy,w,h) -> COCO absolute (x,y,w,h) top-left origin."""
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    bw = w * img_w
    bh = h * img_h
    return [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)]


def normalize_class_name(name):
    """Remove characters that break paths/shells (e.g. 'Fuel?' -> 'Fuel')."""
    return name.replace("?", "").strip()


def build_coco_dict(images, annotations, categories):
    return {
        "info": {"year": 2026, "version": "1.0", "contributor": "Heimdall"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def write_split(entries, split_name, coco_dir, categories):
    coco_images_dir = coco_dir / "images"
    coco_images_dir.mkdir(parents=True, exist_ok=True)

    images_out = []
    annotations_out = []
    ann_id = 1

    for img_id, (img_path, lbl_path) in enumerate(entries, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: could not read {img_path}, skipping")
            continue
        h, w = img.shape[:2]

        images_out.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        dst = coco_images_dir / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)

        content = lbl_path.read_text().strip()
        if not content:
            continue
        for line in content.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            bbox = yolo_to_coco_bbox(cx, cy, bw, bh, w, h)
            area = round(bbox[2] * bbox[3], 2)
            annotations_out.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    out_file = coco_dir / f"{split_name}.json"
    out_file.write_text(json.dumps(build_coco_dict(images_out, annotations_out, categories), indent=2))
    print(f"  Wrote {out_file} ({len(images_out)} images, {len(annotations_out)} annotations)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to Label Studio YOLO export zip")
    parser.add_argument("--out", default="dataset", help="Output root directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    out = Path(args.out)
    raw_dir = out / "raw"
    coco_dir = out / "coco"
    raw_dir.mkdir(parents=True, exist_ok=True)
    coco_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {args.zip} -> {raw_dir}")
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(raw_dir)

    classes_file = raw_dir / "classes.txt"
    raw_classes = [c.strip() for c in classes_file.read_text().splitlines() if c.strip()]
    categories = [{"id": i, "name": normalize_class_name(c)} for i, c in enumerate(raw_classes)]
    print(f"Classes: {[cat['name'] for cat in categories]}")

    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")

    labeled = [
        (f, labels_dir / (f.stem + ".txt"))
        for f in image_files
        if (labels_dir / (f.stem + ".txt")).exists()
    ]
    unlabeled_count = len(image_files) - len(labeled)
    print(f"  Labeled: {len(labeled)}, Unlabeled (skipped): {unlabeled_count}")

    random.seed(args.seed)
    random.shuffle(labeled)
    n_val = max(1, int(len(labeled) * args.val_split))
    val_set = labeled[:n_val]
    train_set = labeled[n_val:]
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}")

    write_split(train_set, "train", coco_dir, categories)
    write_split(val_set, "val", coco_dir, categories)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
