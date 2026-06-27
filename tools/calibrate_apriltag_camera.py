#!/usr/bin/env python3
"""
Calibrates a camera's intrinsics using a checkerboard pattern.
Outputs fx, fy, cx, cy, and distortion coefficients ready to paste into
config/apriltag_layout.json.

Usage:
    python3 calibrate_apriltag_camera.py --device /dev/video4 --squares 9x6 --size 0.025

Arguments:
    --device    V4L2 device path of the AprilTag camera
    --squares   Checkerboard interior corners, WIDTHxHEIGHT  (default 9x6)
    --size      Physical size of one checkerboard square in metres  (default 0.025 = 25mm)
    --width     Capture width   (default 640)
    --height    Capture height  (default 480)

While running:
    SPACE   — capture the current frame (need at least 15 good captures)
    q       — quit and compute calibration

Print a checkerboard from:
    https://calib.io/pages/camera-calibration-pattern-generator
    (choose "Chess Board", set your square size to match --size)
"""

import argparse
import sys
import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="/dev/video4")
    p.add_argument("--squares", default="9x6")
    p.add_argument("--size", type=float, default=0.025)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args()

def main():
    args = parse_args()
    cols, rows = [int(x) for x in args.squares.split("x")]
    sq = args.size

    # 3-D positions of checkerboard corners in board-local space
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq

    obj_pts, img_pts = [], []

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.device}")

    print(f"Checkerboard: {cols}×{rows} corners, {sq*100:.1f} cm squares")
    print("Press SPACE to capture, q to finish and calibrate (need ≥15 captures).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            cv2.putText(display, "Board found — SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(display, "No board detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(display, f"Captures: {len(obj_pts)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_pts.append(objp)
            img_pts.append(corners2)
            print(f"  Captured frame {len(obj_pts)}")

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_pts) < 10:
        sys.exit(f"Only {len(obj_pts)} captures — need at least 10. Re-run and capture more views.")

    print(f"\nCalibrating with {len(obj_pts)} frames…")
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, (args.width, args.height), None, None)

    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    k1, k2, p1, p2, k3 = [float(x) for x in dist[0][:5]]

    print(f"\nRMS reprojection error: {rms:.4f} px  (< 1.0 is good, < 0.5 is great)")
    print()
    print("── Paste into config/apriltag_layout.json ──────────────────────")
    print(f'  "camera": {{')
    print(f'    "device": "{args.device}",')
    print(f'    "width":  {args.width},')
    print(f'    "height": {args.height},')
    print(f'    "fps":    10,')
    print(f'    "fx": {fx:.4f},')
    print(f'    "fy": {fy:.4f},')
    print(f'    "cx": {cx:.4f},')
    print(f'    "cy": {cy:.4f},')
    print(f'    "k1": {k1:.6f},')
    print(f'    "k2": {k2:.6f},')
    print(f'    "p1": {p1:.6f},')
    print(f'    "p2": {p2:.6f},')
    print(f'    "k3": {k3:.6f}')
    print(f'  }},')
    print("─────────────────────────────────────────────────────────────────")

    if rms > 1.5:
        print("\nWARNING: high reprojection error. Try:")
        print("  • More captures (30+) from varied angles")
        print("  • Hold the board flat — no flex")
        print("  • Avoid motion blur (use bright lighting)")

if __name__ == "__main__":
    main()
