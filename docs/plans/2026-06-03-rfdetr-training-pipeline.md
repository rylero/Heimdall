# RF-DETR Nano Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a complete training/export/INT8-calibration pipeline for RF-DETR Nano in `training/`.  
**Architecture:** Seven independent Python scripts + a shell script; no C++ changes. Scripts run sequentially (each stage's output feeds the next). Calibration scripts are Jetson-side; all others run on Windows.  
**Tech Stack:** Python 3.10+, rfdetr>=1.1.0, PyTorch, OpenCV, NumPy, onnx; TensorRT + pycuda on Jetson.  
**Assumptions:** Dataset zip at `C:\Users\ryan\Downloads\project-5-at-2026-01-17-20-11-0f80986e.zip`. rfdetr package uses COCO JSON as training input. Training runs on this Windows machine. `gen_calib_cache.py` and `run_trtexec_int8.sh` are written but not executed here — they run on the Jetson after copying the ONNX and calibration images.

---

## File Structure

| File | Responsibility |
|---|---|
| `training/requirements.txt` | Python deps for Windows training environment |
| `training/scripts/01_prepare_dataset.py` | Extract zip, YOLO→COCO JSON, 80/20 split |
| `training/scripts/02_train.py` | Fine-tune RFDETRNano at 480px, save best.pt |
| `training/scripts/03_export_onnx.py` | Export best.pt → ONNX, print tensor shapes |
| `training/scripts/04_prepare_calibration.py` | Preprocess 50 train images → .npy calib files |
| `training/calibration/gen_calib_cache.py` | IInt8EntropyCalibrator2 → calib.cache (Jetson) |
| `training/calibration/run_trtexec_int8.sh` | trtexec INT8 conversion script (Jetson) |

---

### Task 1: Create directory structure and requirements.txt

**Files:**
- Create: `training/scripts/.gitkeep`
- Create: `training/calibration/.gitkeep`
- Create: `training/dataset/raw/.gitkeep`
- Create: `training/dataset/coco/.gitkeep`
- Create: `training/models/checkpoints/.gitkeep`
- Create: `training/requirements.txt`

**Security flag:** `none`

- [ ] **Step 1: Create directories and requirements.txt**

```bash
mkdir -p training/scripts training/calibration training/dataset/raw training/dataset/coco training/models/checkpoints
touch training/scripts/.gitkeep training/calibration/.gitkeep training/dataset/raw/.gitkeep training/dataset/coco/.gitkeep training/models/checkpoints/.gitkeep
```

Create `training/requirements.txt`:

```
rfdetr>=1.1.0
torch>=2.0.0
torchvision
opencv-python
numpy
Pillow
onnx
```

- [ ] **Step 2: Verify**

Run: `ls training/scripts training/calibration training/dataset/raw training/dataset/coco training/models/checkpoints`  
Expected: all five directories exist with `.gitkeep` files.

Run: `cat training/requirements.txt`  
Expected: 7 lines, rfdetr first.

- [ ] **Step 3: Commit**

```bash
git add training/requirements.txt training/scripts/.gitkeep training/calibration/.gitkeep training/dataset/raw/.gitkeep training/dataset/coco/.gitkeep training/models/checkpoints/.gitkeep
git commit -m "feat(training): scaffold training/ directory structure and requirements"
```

---

### Task 2: Write 01_prepare_dataset.py

**Files:**
- Create: `training/scripts/01_prepare_dataset.py`

**Security flag:** `none`

- [ ] **Step 1: Write script**

Create `training/scripts/01_prepare_dataset.py`:

```python
"""
Extract Label Studio YOLO export zip and convert to COCO JSON format.

Usage:
  python training/scripts/01_prepare_dataset.py \
    --zip "C:/Users/ryan/Downloads/project-5-at-2026-01-17-20-11-0f80986e.zip" \
    --out training/dataset
"""
import argparse
import json
import os
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


def write_split(entries, split_name, coco_dir, images_src_dir, categories):
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
    parser.add_argument("--out", default="training/dataset", help="Output root directory")
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

    write_split(train_set, "train", coco_dir, images_dir, categories)
    write_split(val_set, "val", coco_dir, images_dir, categories)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run against the real dataset to verify**

Run:
```
cd C:\Users\ryan\Dev\Heimdall
python training/scripts/01_prepare_dataset.py --zip "C:\Users\ryan\Downloads\project-5-at-2026-01-17-20-11-0f80986e.zip" --out training/dataset
```

Expected output (exact numbers):
```
Extracting ... -> training/dataset/raw
Classes: ['Fuel']
Found 60 images
  Labeled: 54, Unlabeled (skipped): 6
  Train: 44, Val: 10
  Wrote training/dataset/coco/train.json (44 images, ... annotations)
  Wrote training/dataset/coco/val.json (10 images, ... annotations)
Dataset preparation complete.
```

Also verify JSON structure:
```
python -c "import json; d=json.load(open('training/dataset/coco/train.json')); print('categories:', d['categories']); print('images:', len(d['images'])); print('annotations:', len(d['annotations'])); print('first ann:', d['annotations'][0])"
```
Expected: `categories: [{'id': 0, 'name': 'Fuel'}]`, images count ~44, annotations count ~330, first annotation has keys `id/image_id/category_id/bbox/area/iscrowd`.

- [ ] **Step 3: Commit**

```bash
git add training/scripts/01_prepare_dataset.py
git commit -m "feat(training): add 01_prepare_dataset.py — YOLO→COCO conversion with 80/20 split"
```

---

### Task 3: Write 02_train.py

**Files:**
- Create: `training/scripts/02_train.py`

**Security flag:** `none`

- [ ] **Step 1: Write script**

Create `training/scripts/02_train.py`:

```python
"""
Fine-tune RF-DETR Nano on the prepared COCO dataset.

Usage:
  python training/scripts/02_train.py \
    --data training/dataset/coco \
    --epochs 100 \
    --batch 8 \
    --imgsz 480 \
    --out training/models

Run 01_prepare_dataset.py first.
"""
import argparse
import shutil
import sys
from pathlib import Path


def get_model_class():
    """Return RFDETRNano if available; fall back to RFDETRBase with a warning."""
    try:
        from rfdetr import RFDETRNano
        print("Using RFDETRNano")
        return RFDETRNano
    except ImportError:
        pass
    try:
        from rfdetr import RFDETRBase
        print(
            "WARNING: RFDETRNano not found in the installed rfdetr version.\n"
            "Falling back to RFDETRBase. Upgrade with: pip install rfdetr>=1.1.0"
        )
        return RFDETRBase
    except ImportError:
        print("ERROR: rfdetr not installed. Run: pip install rfdetr>=1.1.0")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/dataset/coco",
                        help="COCO dataset dir with train.json, val.json, images/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--out", default="training/models")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for required in ["train.json", "val.json", "images"]:
        if not (data_dir / required).exists():
            print(f"ERROR: {data_dir / required} not found. Run 01_prepare_dataset.py first.")
            sys.exit(1)

    ModelClass = get_model_class()

    model = ModelClass(
        num_classes=1,
        resolution=args.imgsz,
    )

    print(f"Training: epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}")
    print(f"Dataset: {data_dir}")

    model.train(
        dataset_dir=str(data_dir),
        epochs=args.epochs,
        batch_size=args.batch,
        output_dir=str(checkpoints_dir),
        save_best=True,
    )

    # rfdetr saves the best checkpoint as 'best.pt' inside output_dir.
    # If the file name differs across versions, fall back to the latest .pt.
    best_candidates = (
        list(checkpoints_dir.glob("best.pt"))
        + list(checkpoints_dir.glob("*best*.pt"))
    )
    if best_candidates:
        best_src = sorted(best_candidates)[-1]
    else:
        all_pts = sorted(checkpoints_dir.glob("*.pt"))
        if not all_pts:
            print("WARNING: No checkpoint files found after training.")
            return
        best_src = all_pts[-1]
        print(f"WARNING: No 'best' checkpoint found; using latest: {best_src.name}")

    best_dst = out_dir / "best.pt"
    shutil.copy2(best_src, best_dst)
    print(f"Best checkpoint -> {best_dst}")
    print("Training complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax and flag check (rfdetr not required to be installed)**

Run:
```
python -c "import ast; ast.parse(open('training/scripts/02_train.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

Run (verify --help works without rfdetr installed):
```
python training/scripts/02_train.py --help
```
Expected: prints usage without ImportError.

- [ ] **Step 3: Commit**

```bash
git add training/scripts/02_train.py
git commit -m "feat(training): add 02_train.py — RFDETRNano fine-tune with Nano/Base fallback"
```

---

### Task 4: Write 03_export_onnx.py

**Files:**
- Create: `training/scripts/03_export_onnx.py`

**Security flag:** `none`

- [ ] **Step 1: Write script**

Create `training/scripts/03_export_onnx.py`:

```python
"""
Export trained RF-DETR checkpoint to ONNX and print tensor shapes.

Usage:
  python training/scripts/03_export_onnx.py \
    --checkpoint training/models/best.pt \
    --imgsz 480 \
    --out training/models/rfdetr_nano_480.onnx

After export, compare printed tensor names/shapes against
src/models/bbox_parser/rfdetr_parser.cpp before deploying.
"""
import argparse
import sys
from pathlib import Path


def get_model_class():
    try:
        from rfdetr import RFDETRNano
        print("Using RFDETRNano for export")
        return RFDETRNano
    except ImportError:
        pass
    try:
        from rfdetr import RFDETRBase
        print("WARNING: RFDETRNano not found; exporting with RFDETRBase")
        return RFDETRBase
    except ImportError:
        print("ERROR: rfdetr not installed. Run: pip install rfdetr>=1.1.0")
        sys.exit(1)


def print_onnx_info(onnx_path):
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        print("\n=== ONNX Tensor Shapes ===")
        print("Verify these against src/models/bbox_parser/rfdetr_parser.cpp")
        print("Inputs:")
        for inp in m.graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {dims}")
        print("Outputs:")
        for out in m.graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {dims}")
        print("===========================\n")
    except ImportError:
        print("(Install 'onnx' to see tensor shapes: pip install onnx)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/models/best.pt")
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--out", default="training/models/rfdetr_nano_480.onnx")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: {ckpt} not found. Run 02_train.py first.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ModelClass = get_model_class()
    model = ModelClass(pretrain_weights=str(ckpt), resolution=args.imgsz)

    print(f"Exporting ONNX -> {out_path}")
    model.export(format="onnx", output_path=str(out_path))

    if out_path.exists():
        print_onnx_info(out_path)
        print(f"Export complete: {out_path}")
    else:
        print(f"WARNING: Export method did not produce {out_path}.")
        print("Check rfdetr documentation for the correct export API in your installed version.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run:
```
python -c "import ast; ast.parse(open('training/scripts/03_export_onnx.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

Run:
```
python training/scripts/03_export_onnx.py --help
```
Expected: prints usage, no crash.

- [ ] **Step 3: Commit**

```bash
git add training/scripts/03_export_onnx.py
git commit -m "feat(training): add 03_export_onnx.py — ONNX export with tensor shape verification"
```

---

### Task 5: Write 04_prepare_calibration.py

**Files:**
- Create: `training/scripts/04_prepare_calibration.py`

**Security flag:** `none`

- [ ] **Step 1: Write script**

Create `training/scripts/04_prepare_calibration.py`:

```python
"""
Preprocess training images for TensorRT INT8 calibration.

Outputs float32 CHW numpy arrays to training/calibration/images/.
Preprocessing MUST match the runtime GStreamer/DeepStream pipeline:
  - Resize to imgsz x imgsz (bilinear)
  - BGR -> RGB
  - Normalize: (pixel/255 - mean) / std  with ImageNet mean/std
  - Layout: CHW, dtype float32

Usage:
  python training/scripts/04_prepare_calibration.py \
    --data training/dataset/raw/images \
    --split training/dataset/coco/train.json \
    --out training/calibration \
    --n 50 \
    --imgsz 480
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
    parser.add_argument("--data", default="training/dataset/raw/images")
    parser.add_argument("--split", default="training/dataset/coco/train.json",
                        help="COCO train.json — limits candidates to train split only")
    parser.add_argument("--out", default="training/calibration")
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
```

- [ ] **Step 2: Run against the extracted dataset to verify**

Run:
```
python training/scripts/04_prepare_calibration.py \
  --data training/dataset/raw/images \
  --split training/dataset/coco/train.json \
  --out training/calibration \
  --n 50 \
  --imgsz 480
```

Expected output:
```
Using 44 images from train split
Preprocessing 44 images -> training/calibration/images
Saved 44 calibration tensors
Shape per tensor: (3, 480, 480), float32
List file: training/calibration/calib_list.txt
```
(44 because only 44 train images exist; `--n 50` is capped automatically)

Verify .npy shape:
```
python -c "import numpy as np, glob; f=sorted(glob.glob('training/calibration/images/*.npy'))[0]; a=np.load(f); print('shape:', a.shape, 'dtype:', a.dtype, 'min:', a.min().round(3), 'max:', a.max().round(3))"
```
Expected: `shape: (3, 480, 480) dtype: float32` with min around -2.1 and max around 2.6 (ImageNet normalization range).

- [ ] **Step 3: Commit**

```bash
git add training/scripts/04_prepare_calibration.py
git commit -m "feat(training): add 04_prepare_calibration.py — INT8 calibration image preprocessing"
```

---

### Task 6: Write calibration/gen_calib_cache.py

**Files:**
- Create: `training/calibration/gen_calib_cache.py`

**Security flag:** `none`

- [ ] **Step 1: Write script**

Create `training/calibration/gen_calib_cache.py`:

```python
"""
Generate TensorRT INT8 calibration cache from preprocessed .npy tensors.

Run on the Jetson BEFORE run_trtexec_int8.sh.
Requires: tensorrt, pycuda (both ship with JetPack).

Usage (from training/calibration/ on Jetson):
  python gen_calib_cache.py \
    --onnx ../models/rfdetr_nano_480.onnx \
    --list calib_list.txt \
    --cache calib.cache \
    --imgsz 480
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="../models/rfdetr_nano_480.onnx")
    parser.add_argument("--list", default="calib_list.txt", dest="calib_list")
    parser.add_argument("--cache", default="calib.cache")
    parser.add_argument("--imgsz", type=int, default=480)
    args = parser.parse_args()

    try:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import numpy as np
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Run this script on the Jetson — TensorRT and pycuda are included with JetPack.")
        sys.exit(1)

    list_file = Path(args.calib_list)
    if not list_file.exists():
        print(f"ERROR: {list_file} not found. Copy calibration/ from Windows training machine.")
        sys.exit(1)
    image_paths = [p.strip() for p in list_file.read_text().splitlines() if p.strip()]
    print(f"Calibration images: {len(image_paths)}")

    class Int8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, image_paths, cache_file, imgsz):
            super().__init__()
            self.image_paths = image_paths
            self.cache_file = cache_file
            self.imgsz = imgsz
            self.idx = 0
            nbytes = 3 * imgsz * imgsz * 4  # float32 bytes
            self.device_buf = cuda.mem_alloc(nbytes)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(self.image_paths):
                return None
            tensor = np.load(self.image_paths[self.idx]).astype(np.float32)
            # TRT expects (1, C, H, W) contiguous
            tensor = np.ascontiguousarray(tensor[np.newaxis])
            cuda.memcpy_htod(self.device_buf, tensor)
            self.idx += 1
            if self.idx % 10 == 0 or self.idx == len(self.image_paths):
                print(f"  {self.idx}/{len(self.image_paths)} images calibrated...")
            return [self.device_buf]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                print(f"Loading existing cache: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            print(f"Calibration cache written: {self.cache_file}")

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found. Copy rfdetr_nano_480.onnx from Windows machine.")
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    calibrator = Int8Calibrator(image_paths, args.cache, args.imgsz)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not onnx_parser.parse(f.read()):
            for i in range(onnx_parser.num_errors):
                print(onnx_parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    print("Building calibration engine (may take several minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("ERROR: TRT engine build failed.")
        sys.exit(1)

    print(f"\nCalibration cache ready: {args.cache}")
    print("Next: run ./run_trtexec_int8.sh")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax (TRT not available on Windows)**

Run:
```
python -c "import ast; ast.parse(open('training/calibration/gen_calib_cache.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add training/calibration/gen_calib_cache.py
git commit -m "feat(training): add gen_calib_cache.py — TRT IInt8EntropyCalibrator2 for Jetson"
```

---

### Task 7: Write run_trtexec_int8.sh

**Files:**
- Create: `training/calibration/run_trtexec_int8.sh`

**Security flag:** `none`

- [ ] **Step 1: Write script**

Create `training/calibration/run_trtexec_int8.sh`:

```bash
#!/bin/bash
# Convert RF-DETR ONNX to TensorRT INT8 engine using pre-generated calibration cache.
#
# Run from training/calibration/ on Jetson AFTER:
#   1. Copying training/calibration/ (calib images + calib_list.txt) from Windows
#   2. Copying training/models/rfdetr_nano_480.onnx from Windows
#   3. Running:  python gen_calib_cache.py   (generates calib.cache)
#
# trtexec location on JetPack: /usr/src/tensorrt/bin/trtexec
# Add to PATH if needed: export PATH=$PATH:/usr/src/tensorrt/bin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
ONNX="$MODELS_DIR/rfdetr_nano_480.onnx"
ENGINE="$MODELS_DIR/rfdetr_nano_480_int8.trt"
CALIB_CACHE="$SCRIPT_DIR/calib.cache"

if [ ! -f "$ONNX" ]; then
    echo "ERROR: $ONNX not found."
    echo "Copy rfdetr_nano_480.onnx from the Windows training machine."
    exit 1
fi

if [ ! -f "$CALIB_CACHE" ]; then
    echo "ERROR: $CALIB_CACHE not found."
    echo "Run: python gen_calib_cache.py"
    exit 1
fi

echo "=== TensorRT INT8 Conversion ==="
echo "ONNX:   $ONNX"
echo "Cache:  $CALIB_CACHE"
echo "Engine: $ENGINE"
echo ""

trtexec \
    --onnx="$ONNX" \
    --int8 \
    --fp16 \
    --loadCalib="$CALIB_CACHE" \
    --saveEngine="$ENGINE" \
    --workspace=4096

echo ""
echo "Done: $ENGINE"
echo ""
echo "Deploy this .trt engine in the Heimdall DeepStream pipeline."
echo "Update the nvinfer config to point at: rfdetr_nano_480_int8.trt"
```

- [ ] **Step 2: Verify file**

Run:
```
bash -n training/calibration/run_trtexec_int8.sh && echo "syntax OK"
```
Expected: `syntax OK`

Run:
```
cat training/calibration/run_trtexec_int8.sh | head -5
```
Expected: first line is `#!/bin/bash`

- [ ] **Step 3: Commit**

```bash
git add training/calibration/run_trtexec_int8.sh
git commit -m "feat(training): add run_trtexec_int8.sh — Jetson trtexec INT8 conversion script"
```

---

## Post-Plan Checklist

After all tasks complete, run these to confirm the pipeline is end-to-end ready:

```bash
# Verify all files exist
ls training/scripts/01_prepare_dataset.py \
   training/scripts/02_train.py \
   training/scripts/03_export_onnx.py \
   training/scripts/04_prepare_calibration.py \
   training/calibration/gen_calib_cache.py \
   training/calibration/run_trtexec_int8.sh \
   training/requirements.txt

# Verify calibration data was produced by Task 5
python -c "
import glob, numpy as np
files = glob.glob('training/calibration/images/*.npy')
print(f'Calibration tensors: {len(files)}')
a = np.load(files[0])
print(f'Shape: {a.shape}, dtype: {a.dtype}')
assert a.shape == (3, 480, 480), 'Wrong shape!'
assert a.dtype == np.float32, 'Wrong dtype!'
print('PASS')
"

# Verify COCO JSON is valid
python -c "
import json
for split in ['train', 'val']:
    d = json.load(open(f'training/dataset/coco/{split}.json'))
    assert 'images' in d and 'annotations' in d and 'categories' in d
    assert d['categories'][0]['name'] == 'Fuel'
    print(f'{split}: {len(d[\"images\"])} images, {len(d[\"annotations\"])} annotations - OK')
"
```

## Deployment Notes

After training completes on Windows and the INT8 engine is built on the Jetson:

1. Verify ONNX tensor names/shapes printed by `03_export_onnx.py` match `src/models/bbox_parser/rfdetr_parser.cpp`.
2. Copy `rfdetr_nano_480_int8.trt` to the Jetson's Heimdall deployment directory.
3. Update the DeepStream `nvinfer` config to reference the new engine file.
