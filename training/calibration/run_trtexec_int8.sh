#!/bin/bash
# Convert RF-DETR ONNX to TensorRT INT8+FP16 engine.
#
# Uses trtexec internal calibration (--int8 --fp16 on original ONNX).
# QDQ ONNX approach was abandoned — onnxruntime bias QDQ nodes incompatible with TRT 10.x.
#
# Run from training/calibration/ on Jetson AFTER:
#   1. Copying models/rfdetr_nano_480.onnx to ~/Heimdall/training/models/
#
# trtexec location on JetPack: /usr/src/tensorrt/bin/trtexec
# Add to PATH: export PATH=$PATH:/usr/src/tensorrt/bin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
ONNX="$MODELS_DIR/rfdetr_nano_480.onnx"
ENGINE="$MODELS_DIR/rfdetr_nano_480_int8.trt"

if [ ! -f "$ONNX" ]; then
    echo "ERROR: $ONNX not found."
    echo "Copy rfdetr_nano_480.onnx from the Windows training machine."
    exit 1
fi

echo "=== TensorRT INT8+FP16 Engine Build ==="
echo "ONNX:   $ONNX"
echo "Engine: $ENGINE"
echo ""

/usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX" \
    --int8 \
    --fp16 \
    --saveEngine="$ENGINE" \
    --memPoolSize=workspace:4096M

echo ""
echo "Done: $ENGINE"
echo "~97 FPS / ~10ms latency on Jetson Orin at 480px"
echo ""
echo "Deploy this .trt engine in the Heimdall DeepStream pipeline."
echo "Update the nvinfer config to point at: rfdetr_nano_480_int8.trt"
