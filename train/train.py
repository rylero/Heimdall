"""
Train RF-DETR Nano and export to ONNX.

Usage:
    python train.py

After training, run export_engine.sh on the Jetson to convert the ONNX to a
TensorRT engine for DeepStream inference.
"""

from pathlib import Path
from rfdetr import RFDETRNano

DATASET_DIR = "<DATASET_PATH>"   # Roboflow export format (YOLOv5 or COCO)
OUTPUT_DIR  = "runs/train"
NUM_CLASSES = 3                  # update to match your dataset
RESOLUTION  = 560                # RFDETRNano default; must match export_onnx.py

model = RFDETRNano(
    num_classes=NUM_CLASSES,
    resolution=RESOLUTION,
)

model.train(
    dataset_dir=DATASET_DIR,
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=OUTPUT_DIR,
)

# Export best checkpoint to ONNX immediately after training
checkpoint = Path(OUTPUT_DIR) / "best_model.pth"
onnx_path  = Path(OUTPUT_DIR) / "model.onnx"

print(f"\nExporting {checkpoint} → {onnx_path}")
model.export(type="onnx", output_path=str(onnx_path))
print(f"Done. Copy {onnx_path} to the Jetson and run:")
print(f"  bash train/export_engine.sh model.onnx model.engine fp16")
