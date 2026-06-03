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
