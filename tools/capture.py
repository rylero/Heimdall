#!/usr/bin/env python3
"""
Frame capture helper — run this on the Jetson to collect calibration images.

Usage:
  python tools/capture.py --device /dev/video0 --out /tmp/calib_cam0/
  python tools/capture.py --device /dev/video2 --out /tmp/calib_cam1/

  SPACE — save current frame
  Q     — quit

Saved files are named 000.png, 001.png, ... in the output directory.
Transfer to the laptop with:
  scp jetson:/tmp/calib_cam0/*.png ~/calib/cam0/
"""

import argparse
import sys
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture calibration frames from a camera")
    ap.add_argument("--device", default="/dev/video0", help="Camera device path or index")
    ap.add_argument("--out",    required=True,          help="Output directory for saved frames")
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Accept numeric index or /dev path
    device = int(args.device) if args.device.isdigit() else args.device
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    idx = 0
    print(f"Capturing from {args.device} → {out}")
    print("  SPACE = save frame   Q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed — retrying")
            continue

        cv2.imshow("capture  [SPACE=save  Q=quit]", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            fname = out / f"{idx:03d}.png"
            cv2.imwrite(str(fname), frame)
            print(f"  Saved {fname}")
            idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {idx} frames saved to {out}")


if __name__ == "__main__":
    main()
