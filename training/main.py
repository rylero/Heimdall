"""
Fine-tune YOLO26n on the FRC fuel detection dataset, then export to ONNX.

Run from repo root:
    cd training
    uv run python main.py

Outputs:
    runs/train/yolo26n_fuel/weights/best.pt   <- best checkpoint
    ../models/yolo26n_fuel_640.onnx           <- ready for trtexec
"""

import os
from pathlib import Path
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["MLFLOW_TRACKING_URI"] = ""

from ultralytics import YOLO
import ultralytics.utils.callbacks.mlflow as _mlflow
_mlflow.callbacks = {}  # disable mlflow entirely

DATASET  = Path(__file__).parent / "dataset" / "data.yaml"
RUNS_DIR = Path(__file__).parent / "runs"
ONNX_OUT = Path(__file__).parent.parent / "models" / "yolo26n_fuel_640.onnx"


def train():
    model = YOLO("yolo26n.pt")   # pretrained COCO weights; downloads if not cached

    model.train(
        data=str(DATASET),
        epochs=100,
        imgsz=640,
        batch=16,
        device="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
        project=str(RUNS_DIR / "train"),
        name="yolo26n_fuel",
        exist_ok=True,
        patience=20,          # early stop if no improvement for 20 epochs
        save_period=10,
        cos_lr=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
    )

    return RUNS_DIR / "train" / "yolo26n_fuel" / "weights" / "best.pt"


def export(weights: Path):
    model = YOLO(str(weights))
    model.export(
        format="onnx",
        imgsz=640,
        dynamic=True,
        simplify=True,
        end2end=False,   # traditional output for DeepStream bbox parser
        opset=20,
    )
    # Ultralytics writes the ONNX next to the .pt file — move it to models/
    src = weights.with_suffix(".onnx")
    ONNX_OUT.parent.mkdir(exist_ok=True)
    src.rename(ONNX_OUT)
    print(f"Exported → {ONNX_OUT}")


if __name__ == "__main__":
    weights = train()
    export(weights)
