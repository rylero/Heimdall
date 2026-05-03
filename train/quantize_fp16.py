"""
Convert an FP32 ONNX model to FP16 to reduce file size (~2x).
The resulting model is fully compatible with export_engine.sh (trtexec --fp16).

Usage:
    python quantize_fp16.py --input models/inference_model.onnx \
                            --output models/inference_model.onnx
"""

import argparse
from pathlib import Path

import onnx
from onnxconverter_common import float16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    print(f"Loading {src}  ({src.stat().st_size / 1e6:.1f} MB)")
    model = onnx.load(str(src))

    print("Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(dst))
    print(f"Saved {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
