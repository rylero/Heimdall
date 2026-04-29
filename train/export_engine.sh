#!/bin/bash
# export_engine.sh — run on Jetson (inside Docker or native JetPack)
#
# Converts an ONNX model to a TensorRT .engine file.
#
# Usage:
#   ./export_engine.sh <input.onnx> <output.engine> [fp16|int8]
#
# Precision options:
#   fp16  — half precision, no calibration needed (default, recommended for first run)
#   int8  — 8-bit quantization, ~2x faster than fp16 but requires calibration data
#
# Examples:
#   ./export_engine.sh model.onnx model_fp16.engine fp16
#   ./export_engine.sh model.onnx model_int8.engine int8
#
# After export, copy the .engine file to the Heimdall models/ directory and
# update config/infer_rfdetr.txt with the correct ENGINE_PATH.

set -euo pipefail

ONNX_PATH="${1:?Usage: $0 <input.onnx> <output.engine> [fp16|int8]}"
ENGINE_PATH="${2:?Usage: $0 <input.onnx> <output.engine> [fp16|int8]}"
PRECISION="${3:-fp16}"

TRTEXEC=/usr/src/tensorrt/bin/trtexec

if [ ! -f "$TRTEXEC" ]; then
    echo "Error: trtexec not found at $TRTEXEC"
    echo "This script must run on the Jetson inside the DeepStream Docker container."
    exit 1
fi

if [ ! -f "$ONNX_PATH" ]; then
    echo "Error: ONNX file not found: $ONNX_PATH"
    exit 1
fi

echo "========================================"
echo "  RF-DETR → TensorRT Engine Export"
echo "========================================"
echo "  Input:     $ONNX_PATH"
echo "  Output:    $ENGINE_PATH"
echo "  Precision: $PRECISION"
echo ""

PRECISION_FLAGS=""
case "$PRECISION" in
    fp16)
        PRECISION_FLAGS="--fp16"
        ;;
    int8)
        PRECISION_FLAGS="--int8 --fp16"
        echo "NOTE: INT8 without a calibration file uses implicit quantization."
        echo "For best INT8 accuracy, provide a calibration cache via --calib=<file>."
        echo ""
        ;;
    *)
        echo "Error: unknown precision '$PRECISION'. Use fp16 or int8."
        exit 1
        ;;
esac

# RF-DETR Nano uses 560x560 by default; Base uses 640x640.
# Adjust --optShapes to match your training resolution.
RESOLUTION="${RFDETR_RESOLUTION:-560}"

echo "Running trtexec (this may take several minutes on first run)..."
echo ""

$TRTEXEC \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    $PRECISION_FLAGS \
    --minShapes=input:1x3x${RESOLUTION}x${RESOLUTION} \
    --optShapes=input:1x3x${RESOLUTION}x${RESOLUTION} \
    --maxShapes=input:1x3x${RESOLUTION}x${RESOLUTION} \
    --verbose

echo ""
echo "========================================"
echo "  Export complete: $ENGINE_PATH"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Copy engine to Heimdall models directory:"
echo "       cp $ENGINE_PATH /app/models/"
echo ""
echo "  2. Verify output layer names match rfdetr_parser.cpp:"
echo "       python3 -c \""
echo "         import tensorrt as trt"
echo "         r = trt.Runtime(trt.Logger(trt.Logger.WARNING))"
echo "         e = r.deserialize_cuda_engine(open('$ENGINE_PATH','rb').read())"
echo "         [print(e.get_binding_name(i), e.get_binding_shape(i)) for i in range(e.num_bindings)]"
echo "       \""
echo ""
echo "  3. Update config/infer_rfdetr.txt:"
echo "       ENGINE_PATH  -> /app/models/$(basename $ENGINE_PATH)"
echo "       network-mode -> $([ '$PRECISION' = 'int8' ] && echo '1 (INT8)' || echo '0 (FP16, set network-mode=0)')"
