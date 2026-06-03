# RF-DETR Nano Training Pipeline — Design Spec

**Date:** 2026-06-03  
**Status:** Approved

---

## Scope

Build a training pipeline inside `training/` that:
1. Converts the Label Studio YOLO export to COCO JSON
2. Fine-tunes RF-DETR Nano from COCO pretrained weights at 480×480
3. Exports the best checkpoint to ONNX
4. Generates INT8 calibration data (preprocessed `.npy` images) and a Jetson-side trtexec conversion script

### Non-Goals
- Multi-class or multi-camera training
- Hyperparameter sweep / AutoML
- Validation metrics dashboard
- Any changes to the Heimdall C++ pipeline

---

## Dataset

- **Source:** `project-5-at-2026-01-17-20-11-0f80986e.zip`
- **Format:** YOLO (Label Studio export) — `images/*.jpg`, `labels/*.txt`, `classes.txt`
- **Stats:** 60 images, 54 labeled, 1 class (`Fuel?`), 369 bounding boxes
- **Split:** 80/20 train/val, `seed=42` → ~43 train / ~11 val
- **Class normalization:** `Fuel?` → `Fuel` in COCO JSON output (avoids path/shell issues)
- **Known constraint:** 60 images is small; relies on COCO pretrained backbone + aggressive augmentation

---

## Directory Layout

```
Heimdall/
  training/
    scripts/
      01_prepare_dataset.py     # extract zip, YOLO→COCO, split
      02_train.py               # fine-tune RFDETRNano at 480px
      03_export_onnx.py         # export best.pt → rfdetr_nano_480.onnx, verify output tensors
      04_prepare_calibration.py # preprocess 50 train images → .npy calib files
    calibration/
      gen_calib_cache.py        # IInt8EntropyCalibrator2, .npy → calib.cache (runs on Jetson)
      run_trtexec_int8.sh       # trtexec --loadCalib command (runs on Jetson)
    dataset/
      raw/                      # extracted zip (images/, labels/, classes.txt)
      coco/                     # train.json, val.json, images/
    models/                     # checkpoints/ + rfdetr_nano_480.onnx
    requirements.txt
```

---

## Data Flow

```
zip
 └─ 01_prepare_dataset.py ──→ dataset/coco/train.json + val.json
                                        │
                               02_train.py ──→ models/checkpoints/ + models/best.pt
                                        │
                              03_export_onnx.py ──→ models/rfdetr_nano_480.onnx
                                        │
                    04_prepare_calibration.py ──→ calibration/images/*.npy (50 images)

[Copy to Jetson: rfdetr_nano_480.onnx + calibration/]
                                        │
                         gen_calib_cache.py ──→ calibration/calib.cache
                                        │
                       run_trtexec_int8.sh ──→ rfdetr_nano_480_int8.trt
```

---

## Script Specifications

### `01_prepare_dataset.py`
- Accepts `--zip <path>` and `--out training/dataset`
- Extracts to `dataset/raw/`
- Reads `classes.txt` → builds `categories` list, normalizes class name (`Fuel?` → `Fuel`)
- Parses each `labels/*.txt` (YOLO: `cls cx cy w h` normalized) → COCO `bbox` (`[x, y, w, h]` absolute pixels)
- Image sizes read from actual files (JPEG headers) for accurate conversion
- Shuffles with `seed=42`, splits 80/20
- Writes `dataset/coco/train.json` and `dataset/coco/val.json`
- Symlinks (or copies) images to `dataset/coco/images/`

### `02_train.py`
- `pip install rfdetr` (documents minimum version in requirements.txt)
- Uses `RFDETRNano` class; if unavailable, prints diagnostic and falls back to `RFDETRBase` with warning
- Config: `imgsz=480`, `epochs=100`, `batch=8` (adjustable via CLI args)
- Saves best checkpoint by val mAP to `models/checkpoints/`
- Copies best checkpoint to `models/best.pt`

### `03_export_onnx.py`
- Loads `models/best.pt`
- Calls rfdetr's native ONNX export at 480×480
- Writes `models/rfdetr_nano_480.onnx`
- Prints ONNX graph input/output tensor names and shapes so user can cross-check against `src/models/bbox_parser/rfdetr_parser.cpp`

### `04_prepare_calibration.py`
- Reads 50 images from `dataset/raw/images/` (train split only)
- Preprocessing: resize 480×480 (bilinear), convert BGR→RGB, normalize with ImageNet mean/std `([0.485,0.456,0.406], [0.229,0.224,0.225])`, transpose HWC→CHW, cast float32
- Saves each as `calibration/images/<stem>.npy`
- Also writes `calibration/calib_list.txt` (one path per line) for the Jetson calibrator

### `calibration/gen_calib_cache.py` (Jetson)
- Implements `IInt8EntropyCalibrator2`
- Reads `.npy` files listed in `calib_list.txt`
- Feeds batches (batch size 1) to TensorRT calibration
- Writes `calibration/calib.cache`
- Requires: `tensorrt`, `pycuda`, `numpy`

### `calibration/run_trtexec_int8.sh` (Jetson)
```bash
trtexec \
  --onnx=../models/rfdetr_nano_480.onnx \
  --int8 \
  --loadCalib=calib.cache \
  --saveEngine=../models/rfdetr_nano_480_int8.trt \
  --workspace=4096 \
  --fp16
```
- `--fp16` included alongside `--int8` (TRT uses INT8 for compute, FP16 for layers INT8 can't handle)
- `--workspace=4096` (4 GB) appropriate for Orin

---

## INT8 Preprocessing Contract

The preprocessing in `04_prepare_calibration.py` **must exactly match** the runtime preprocessing in the GStreamer pipeline. Current assumption: ImageNet mean/std normalization, RGB, CHW, float32. If the Heimdall pipeline uses a different normalization (e.g., `[0,1]` range without mean/std), `04_prepare_calibration.py` must be updated to match before running calibration.

---

## Requirements

**Windows (training):**
```
rfdetr>=1.1.0
torch>=2.0.0
torchvision
opencv-python
numpy
Pillow
```

**Jetson (calibration + conversion):**
- TensorRT 8.x+ (ships with JetPack)
- pycuda
- numpy

---

## Known Constraints / Non-Issues

- **`RFDETRNano` availability:** Class name may differ in older rfdetr versions. Script detects and warns.
- **Small dataset:** 43 training images is below typical. Mitigated by pretrained COCO backbone. Acceptable for FRC game piece detection where appearance is consistent.
- **ONNX tensor shape mismatch:** `03_export_onnx.py` prints tensor shapes; user must manually verify against `rfdetr_parser.cpp` before deploying.
- **Calibration preprocessing mismatch:** Documented above as a user responsibility. INT8 accuracy degrades silently if preprocessing differs.
