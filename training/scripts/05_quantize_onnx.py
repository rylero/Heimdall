"""
Static INT8 quantization using QDQ format (TensorRT 10.x compatible).

Produces a pre-quantized ONNX with QDQ nodes using real calibration images.
TRT 10.x converts QDQ ONNX to INT8 engine without needing a separate calibration step.

Usage (run from training/ directory):
  python scripts/05_quantize_onnx.py

Then on Jetson:
  trtexec --onnx=models/rfdetr_nano_480_qdq.onnx --fp16 --saveEngine=models/rfdetr_nano_480_int8.trt --workspace=4096
"""
import sys
from pathlib import Path

import numpy as np


def main():
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: uv add onnxruntime")
        sys.exit(1)

    onnx_path = Path("models/rfdetr_nano_480.onnx")
    qdq_path = Path("models/rfdetr_nano_480_qdq.onnx")
    calib_list = Path("calibration/calib_list.txt")

    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found. Run 03_export_onnx.py first.")
        sys.exit(1)
    if not calib_list.exists():
        print(f"ERROR: {calib_list} not found. Run 04_prepare_calibration.py first.")
        sys.exit(1)

    calib_dir = calib_list.parent
    image_paths = [
        str((calib_dir / p.strip()).resolve())
        for p in calib_list.read_text().splitlines() if p.strip()
    ]
    print(f"Calibration images: {len(image_paths)}")

    class NpyCalibrationReader(CalibrationDataReader):
        def __init__(self, paths, input_name="input"):
            self.paths = paths
            self.input_name = input_name
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.paths):
                return None
            tensor = np.load(self.paths[self.idx]).astype(np.float32)
            self.idx += 1
            # onnxruntime expects NHWC or NCHW depending on model; RF-DETR uses NCHW
            return {self.input_name: tensor[np.newaxis]}  # (1, C, H, W)

    print(f"Quantizing {onnx_path} -> {qdq_path}")
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(qdq_path),
        calibration_data_reader=NpyCalibrationReader(image_paths),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,          # per-tensor; TRT 10.x handles this better
        reduce_range=False,
        extra_options={
            "ActivationSymmetric": True,   # zero_point=0 (INT8), avoids TRT UINT8 QDQ rejection
            "WeightSymmetric": True,
        },
    )

    if qdq_path.exists():
        size_mb = qdq_path.stat().st_size / 1e6
        print(f"QDQ ONNX written: {qdq_path} ({size_mb:.1f} MB)")
        print()
        print("Transfer to Jetson:")
        print(f"  scp models/rfdetr_nano_480_qdq.onnx ryan@192.168.1.176:~/Heimdall/training/models/")
        print()
        print("On Jetson, build TRT engine:")
        print("  trtexec --onnx=rfdetr_nano_480_qdq.onnx --fp16 --saveEngine=rfdetr_nano_480_int8.trt --workspace=4096")
    else:
        print("ERROR: QDQ ONNX was not produced.")
        sys.exit(1)


if __name__ == "__main__":
    main()
