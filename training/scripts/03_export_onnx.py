"""
Export trained RF-DETR checkpoint to ONNX and print tensor shapes.

Usage:
  python training/scripts/03_export_onnx.py \
    --checkpoint training/models/best.pt \
    --imgsz 480 \
    --out training/models/rfdetr_nano_480.onnx

After export, compare printed tensor names/shapes against
src/models/bbox_parser/rfdetr_parser.cpp before deploying.
"""
import argparse
import sys
from pathlib import Path


def get_model_class():
    try:
        from rfdetr import RFDETRNano
        print("Using RFDETRNano for export")
        return RFDETRNano
    except ImportError:
        pass
    try:
        from rfdetr import RFDETRBase
        print("WARNING: RFDETRNano not found; exporting with RFDETRBase")
        return RFDETRBase
    except ImportError:
        print("ERROR: rfdetr not installed. Run: pip install rfdetr>=1.1.0")
        sys.exit(1)


def print_onnx_info(onnx_path):
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        print("\n=== ONNX Tensor Shapes ===")
        print("Verify these against src/models/bbox_parser/rfdetr_parser.cpp")
        print("Inputs:")
        for inp in m.graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {dims}")
        print("Outputs:")
        for out in m.graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {dims}")
        print("===========================\n")
    except ImportError:
        print("(Install 'onnx' to see tensor shapes: pip install onnx)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/models/best.pt")
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--out", default="training/models/rfdetr_nano_480.onnx")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: {ckpt} not found. Run 02_train.py first.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ModelClass = get_model_class()
    model = ModelClass(pretrain_weights=str(ckpt), resolution=args.imgsz)

    print(f"Exporting ONNX -> {out_path}")
    model.export(format="onnx", output_path=str(out_path))

    if out_path.exists():
        print_onnx_info(out_path)
        print(f"Export complete: {out_path}")
    else:
        print(f"WARNING: Export method did not produce {out_path}.")
        print("Check rfdetr documentation for the correct export API in your installed version.")


if __name__ == "__main__":
    main()
