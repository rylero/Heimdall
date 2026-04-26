"""
Export a trained RF-DETR checkpoint to ONNX for TensorRT deployment.

Usage:
    python export_onnx.py --checkpoint runs/train/best_model.pth \
                          --num-classes 3 \
                          --output model.onnx

The exported ONNX will have:
    Input:  images  [1, 3, resolution, resolution]
    Output: pred_logits  [1, num_queries, num_classes]
            pred_boxes   [1, num_queries, 4]  (cx, cy, w, h, normalized 0..1)

These output names are what rfdetr_parser.cpp expects in the DeepStream bbox parser.
Verify after export with:
    python -c "import onnx; m=onnx.load('model.onnx'); [print(o.name) for o in m.graph.output]"
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export RF-DETR checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained .pth checkpoint (e.g. runs/train/best_model.pth)",
    )
    parser.add_argument(
        "--num-classes", type=int, required=True,
        help="Number of object classes the model was trained on",
    )
    parser.add_argument(
        "--output", default="model.onnx",
        help="Output ONNX file path (default: model.onnx)",
    )
    parser.add_argument(
        "--resolution", type=int, default=560,
        help="Input resolution in pixels — must match training resolution. "
             "RFDETRNano default is 560, RFDETRBase default is 640.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading RF-DETR Nano from {checkpoint}")
    print(f"  num_classes={args.num_classes}  resolution={args.resolution}")

    from rfdetr import RFDETRNano

    model = RFDETRNano(
        pretrain_weights=str(checkpoint),
        num_classes=args.num_classes,
        resolution=args.resolution,
    )

    output = Path(args.output)
    print(f"Exporting to {output} ...")

    # rfdetr's built-in ONNX export. Saves to the specified path.
    model.export(type="onnx", output_path=str(output))

    print(f"\nExport complete: {output}")
    print("\nVerify output layer names (must match rfdetr_parser.cpp constants):")
    print(f"  python -c \"import onnx; m=onnx.load('{output}'); [print(o.name) for o in m.graph.output]\"")
    print("\nExpected: pred_logits, pred_boxes")
    print("If names differ, update LOGITS_LAYER / BOXES_LAYER in src/models/bbox_parser/rfdetr_parser.cpp")


if __name__ == "__main__":
    main()
