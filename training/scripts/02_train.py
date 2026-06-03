"""
Fine-tune RF-DETR Nano on the prepared COCO dataset.

Usage (run from training/ directory):
  python scripts/02_train.py --epochs 100 --batch 8

Run scripts/01_prepare_dataset.py first.
"""
import argparse
import shutil
import sys
from pathlib import Path


def get_model_class():
    """Return RFDETRNano if available; fall back to RFDETRBase with a warning."""
    try:
        from rfdetr import RFDETRNano
        print("Using RFDETRNano")
        return RFDETRNano
    except ImportError:
        pass
    try:
        from rfdetr import RFDETRBase
        print(
            "WARNING: RFDETRNano not found in the installed rfdetr version.\n"
            "Falling back to RFDETRBase. Upgrade with: pip install rfdetr>=1.1.0"
        )
        return RFDETRBase
    except ImportError:
        print("ERROR: rfdetr not installed. Run: pip install rfdetr>=1.1.0")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/coco",
                        help="COCO dataset dir with train.json, val.json, images/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for required in ["train/_annotations.coco.json", "valid/_annotations.coco.json"]:
        if not (data_dir / required).exists():
            print(f"ERROR: {data_dir / required} not found. Run 01_prepare_dataset.py first.")
            sys.exit(1)

    ModelClass = get_model_class()

    model = ModelClass(
        num_classes=1,
        resolution=args.imgsz,
    )

    print(f"Training: epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}")
    print(f"Dataset: {data_dir}")

    model.train(
        dataset_dir=str(data_dir),
        epochs=args.epochs,
        batch_size=args.batch,
        output_dir=str(checkpoints_dir),
        save_best=True,
    )

    # rfdetr saves the best checkpoint as 'best.pt' inside output_dir.
    # If the file name differs across versions, fall back to the latest .pt.
    best_candidates = (
        list(checkpoints_dir.glob("best.pt"))
        + list(checkpoints_dir.glob("*best*.pt"))
    )
    if best_candidates:
        best_src = sorted(best_candidates)[-1]
    else:
        all_pts = sorted(checkpoints_dir.glob("*.pt"))
        if not all_pts:
            print("WARNING: No checkpoint files found after training.")
            return
        best_src = all_pts[-1]
        print(f"WARNING: No 'best' checkpoint found; using latest: {best_src.name}")

    best_dst = out_dir / "best.pt"
    shutil.copy2(best_src, best_dst)
    print(f"Best checkpoint -> {best_dst}")
    print("Training complete.")


if __name__ == "__main__":
    main()
