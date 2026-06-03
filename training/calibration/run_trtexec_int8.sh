#!/bin/bash
# Convert pre-quantized QDQ ONNX to TensorRT INT8 engine.
#
# TRT 10.x workflow: quantize on Windows (05_quantize_onnx.py) → transfer QDQ ONNX → build engine here.
# No separate calibration step needed — quantization is baked into the QDQ ONNX.
#
# Run from training/calibration/ on Jetson AFTER:
#   1. Running scripts/05_quantize_onnx.py on Windows
#   2. Copying models/rfdetr_nano_480_qdq.onnx to this Jetson
#
# trtexec location on JetPack: /usr/src/tensorrt/bin/trtexec
# Add to PATH if needed: export PATH=$PATH:/usr/src/tensorrt/bin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
QDQ_ONNX="$MODELS_DIR/rfdetr_nano_480_qdq.onnx"
ENGINE="$MODELS_DIR/rfdetr_nano_480_int8.trt"

if [ ! -f "$QDQ_ONNX" ]; then
    echo "ERROR: $QDQ_ONNX not found."
    echo "Run on Windows: uv run python scripts/05_quantize_onnx.py"
    echo "Then scp models/rfdetr_nano_480_qdq.onnx ryan@192.168.1.176:~/Heimdall/training/models/"
    exit 1
fi

echo "=== TensorRT INT8 Engine Build ==="
echo "QDQ ONNX: $QDQ_ONNX"
echo "Engine:   $ENGINE"
echo ""

trtexec \
    --onnx="$QDQ_ONNX" \
    --fp16 \
    --saveEngine="$ENGINE" \
    --memPoolSize=workspace:4096M

echo ""
echo "Done: $ENGINE"
echo ""
echo "Deploy this .trt engine in the Heimdall DeepStream pipeline."
echo "Update the nvinfer config to point at: rfdetr_nano_480_int8.trt"
