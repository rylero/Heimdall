#!/usr/bin/env python3
"""Interactive ground-plane grid overlay for calibrated cameras.

Projects a regular grid (robot frame: +X forward, +Y left, Z=0) onto a live
camera feed.  All camera parameters and grid settings are adjustable via
trackbars in real time.

Usage:
  grid_overlay.py
  grid_overlay.py --camera 1 --config ../../config/cameras/cam0.jsonc
  grid_overlay.py --camera rtsp://admin:pass@192.168.1.100:554/stream1

Keys:
  S  – save current settings to config/saved.jsonc
  Q  – quit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent

WIN_MAIN = "Grid Overlay"
WIN_CTRL = "Controls"


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def load_jsonc(path: str | Path) -> dict:
    return json.loads(re.sub(r"//.*", "", Path(path).read_text("utf-8")))


def build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def build_dist(k1, k2, p1, p2, k3) -> np.ndarray:
    return np.array([k1, k2, p1, p2, k3], np.float64)


def R_zyz(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """ZYZ Euler: R_z(roll) @ R_y(π + pitch) @ R_z(yaw) → robot→camera rotation.
    yaw=pan (around robot Z), pitch=tilt from vertical toward ground (positive=more forward),
    roll=image rotation (around optical Z).
    pitch=0 → camera Z points straight down; pitch=π/2 → camera Z points forward.
    """
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(np.pi + pitch), np.sin(np.pi + pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cr * cz * cy - sr * sz, -cr * sz * cy - sr * cz, cr * sy],
            [sr * cz * cy + cr * sz, -sr * sz * cy + cr * cz, sr * sy],
            [-sy * cz, sy * sz, cy],
        ],
        np.float64,
    )


def rvec_tvec_from_extrinsics(tx, ty, tz, yaw, pitch, roll):
    R = R_zyz(yaw, pitch, roll)
    rvec = cv2.Rodrigues(R)[0]
    tvec = -R @ np.array([tx, ty, tz], np.float64)
    return rvec, tvec


def grid_points(extent: float, spacing: float):
    xs = np.arange(0, extent + spacing * 0.5, spacing)
    ys = np.arange(-extent, extent + spacing * 0.5, spacing)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
    return pts, xs, ys


def _rotate_code(rotation: int) -> int | None:
    """Map a config rotation (clockwise degrees, 0/90/180/270) to a cv2.rotate code, so the
    preview matches the rotated frame the pipeline feeds nvinfer. Flips are not supported."""
    return {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(rotation)


# ═══════════════════════════════════════════════════════════════════════════
#  Trackbars
# ═══════════════════════════════════════════════════════════════════════════

# Each: (label, config_key, min, max, scale)
#   trackbar_pos = int(round((default - min) / scale))
#   real_value   = min + getTrackbarPos() * scale
TB_DEFS = [
    ("fx", "fx", 0, 2000, 1),
    ("fy", "fy", 0, 2000, 1),
    ("cx", "cx", 0, 2000, 1),
    ("cy", "cy", 0, 2000, 1),
    ("k1*1k", "k1", -1000, 1000, 1),
    ("k2*1k", "k2", -1000, 1000, 1),
    ("p1*1k", "p1", -1000, 1000, 1),
    ("p2*1k", "p2", -1000, 1000, 1),
    ("k3*1k", "k3", -1000, 1000, 1),
    ("tx_cm", "tx", -200, 200, 1),
    ("ty_cm", "ty", -200, 200, 1),
    ("tz_cm", "tz", 0, 500, 1),
    ("yaw_deg", "yaw", -180, 180, 1),
    ("pitch_deg", "pitch", -180, 180, 1),
    ("roll_deg", "roll", -180, 180, 1),
    ("range_m", "range", 1, 50, 1),
    ("sp_cm", "sp", 10, 500, 1),
    ("units", "units", 0, 1, 1),
]


def init_trackbars(cfg: dict):
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 400, 600)
    K = cfg["intrinsics"]
    D = K["distortion"]
    E = cfg["extrinsics"]

    defs = {
        "fx": K["fx"],
        "fy": K["fy"],
        "cx": K["cx"],
        "cy": K["cy"],
        "k1": D["k1"] * 1000,
        "k2": D["k2"] * 1000,
        "p1": D["p1"] * 1000,
        "p2": D["p2"] * 1000,
        "k3": D["k3"] * 1000,
        "tx": E["tx"] * 100,
        "ty": E["ty"] * 100,
        "tz": E["tz"] * 100,
        "yaw": np.degrees(E["yaw"]),
        "pitch": np.degrees(E["pitch"]),
        "roll": np.degrees(E["roll"]),
        "range": 5,
        "sp": 20,
        "units": 0,
    }

    for label, key, lo, hi, sc in TB_DEFS:
        d = defs[key]
        cv2.createTrackbar(
            label,
            WIN_CTRL,
            int(round((d - lo) / sc)),
            int(round((hi - lo) / sc)),
            lambda _: None,
        )

    def _rd(name, lo, sc):
        return lo + cv2.getTrackbarPos(name, WIN_CTRL) * sc

    def read() -> dict:
        return {
            "fx": _rd("fx", 0, 1),
            "fy": _rd("fy", 0, 1),
            "cx": _rd("cx", 0, 1),
            "cy": _rd("cy", 0, 1),
            "k1": _rd("k1*1k", -1000, 1) / 1000,
            "k2": _rd("k2*1k", -1000, 1) / 1000,
            "p1": _rd("p1*1k", -1000, 1) / 1000,
            "p2": _rd("p2*1k", -1000, 1) / 1000,
            "k3": _rd("k3*1k", -1000, 1) / 1000,
            "tx": _rd("tx_cm", -200, 1) / 100,
            "ty": _rd("ty_cm", -200, 1) / 100,
            "tz": _rd("tz_cm", 0, 1) / 100,
            "yaw": np.radians(_rd("yaw_deg", -180, 1)),
            "pitch": np.radians(_rd("pitch_deg", -180, 1)),
            "roll": np.radians(_rd("roll_deg", -180, 1)),
            "extent": _rd("range_m", 1, 1),
            "spacing": _rd("sp_cm", 10, 1) / 100,
            "units": int(_rd("units", 0, 1)),
        }

    return read


# ═══════════════════════════════════════════════════════════════════════════
#  Drawing
# ═══════════════════════════════════════════════════════════════════════════

C_GRID = (64, 200, 64)
C_ORIGIN = (0, 128, 255)
C_LABEL = (220, 220, 220)
C_HUD = (200, 220, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_grid_overlay(frame, pts_2d, visible, nx, ny, xs, ys, v):
    units_m = v["units"] == 0
    scale_lbl = 1.0 if units_m else 3.28084
    unit_str = "m" if units_m else "ft"
    spacing = v["spacing"]
    extent = v["extent"]

    label_step = max(1, int(round(1.0 / spacing)))
    lbl_scale = 0.35
    lbl_thick = 1

    # ── lines ──
    for j in range(ny):
        for i in range(nx - 1):
            if visible[j, i] and visible[j, i + 1]:
                cv2.line(frame, pts_2d[j, i], pts_2d[j, i + 1], C_GRID, 1, cv2.LINE_AA)

    for i in range(nx):
        for j in range(ny - 1):
            if visible[j, i] and visible[j + 1, i]:
                cv2.line(frame, pts_2d[j, i], pts_2d[j + 1, i], C_GRID, 1, cv2.LINE_AA)

    # ── origin ──
    j0 = ny // 2
    i0 = 0
    if visible[j0, i0]:
        ox, oy = pts_2d[j0, i0]
        cv2.circle(frame, (ox, oy), 5, C_ORIGIN, -1)
        cv2.putText(frame, "0", (ox + 6, oy - 6), FONT, 0.45, C_LABEL, 1)

    # ── X-axis labels (along Y ≈ 0) ──
    for i in range(0, nx, label_step):
        if visible[j0, i]:
            val = xs[i] * scale_lbl
            x, y = pts_2d[j0, i]
            cv2.putText(
                frame, f"{val:.1f}", (x + 4, y - 8), FONT, lbl_scale, C_LABEL, lbl_thick
            )

    # ── Y-axis labels (along X ≈ 0) ──
    for j in range(0, ny, label_step):
        if visible[j, i0] and j != j0:
            val = ys[j] * scale_lbl
            x, y = pts_2d[j, i0]
            cv2.putText(
                frame, f"{val:.1f}", (x + 4, y - 8), FONT, lbl_scale, C_LABEL, lbl_thick
            )

    # ── HUD ──
    lines = [
        f"H: {v['tz']:.2f}m  Tilt: {90 - np.degrees(v['pitch']):.0f}deg fwd  Yaw: {np.degrees(v['yaw']):.1f}deg",
        f"Grid: {extent}m / {spacing * 100:.0f}cm  [{unit_str}]",
    ]
    for idx, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 20 + idx * 20), FONT, 0.45, C_HUD, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args():
    ap = argparse.ArgumentParser(description="Interactive camera grid overlay")
    ap.add_argument("--camera", default="0", help="camera index or RTSP URL")
    ap.add_argument("--config", default=None, help="path to JSONC camera config")
    return ap.parse_args()


def resolve_config(path_arg: str | None) -> str:
    if path_arg:
        return path_arg
    candidates = sorted((PROJ / "config" / "cameras").glob("cam*.jsonc"))
    if candidates:
        return str(candidates[0])
    print("No camera config found.  Use --config to specify one.", file=sys.stderr)
    sys.exit(1)


def main():
    args = parse_args()
    cfg_path = resolve_config(args.config)
    cfg = load_jsonc(cfg_path)
    print(f"Loaded config: {cfg_path}")

    # Open camera
    try:
        cam_src = int(args.camera)
    except ValueError:
        cam_src = args.camera

    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        print(f"Failed to open camera: {args.camera}", file=sys.stderr)
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame", file=sys.stderr)
        sys.exit(1)

    h, w = frame.shape[:2]
    rotate_code = _rotate_code(int(cfg.get("rotation", 0)))
    read_tb = init_trackbars(cfg)

    # Undistortion map cache (keyed by trackbar state tuple)
    _umap_cache = {}

    t_prev = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── read controls ──
        v = read_tb()

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        K = build_K(v["fx"], v["fy"], v["cx"], v["cy"])
        dist = build_dist(v["k1"], v["k2"], v["p1"], v["p2"], v["k3"])
        rvec, tvec = rvec_tvec_from_extrinsics(
            v["tx"], v["ty"], v["tz"], v["yaw"], v["pitch"], v["roll"]
        )

        # ── undistort frame ──
        has_dist = np.any(dist != 0)
        if has_dist:
            ck = (
                v["fx"],
                v["fy"],
                v["cx"],
                v["cy"],
                v["k1"],
                v["k2"],
                v["p1"],
                v["p2"],
                v["k3"],
            )
            if ck not in _umap_cache:
                _umap_cache[ck] = cv2.initUndistortRectifyMap(
                    K, dist, None, K, (w, h), cv2.CV_32FC1
                )
                if len(_umap_cache) > 30:
                    _umap_cache.pop(next(iter(_umap_cache)))
            map1, map2 = _umap_cache[ck]
            frame_show = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        else:
            frame_show = frame.copy()

        # ── project grid ──
        pts_3d, xs, ys = grid_points(v["extent"], v["spacing"])
        ny, nx = len(ys), len(xs)

        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
        pts_2d = pts_2d.reshape(ny, nx, 2).astype(np.int32)

        # camera-space Z visibility check
        R = cv2.Rodrigues(rvec)[0]
        P_cam = (R @ pts_3d.T + tvec.reshape(3, 1)).T.reshape(ny, nx, 3)
        visible = P_cam[:, :, 2] > 0.1

        # ── draw ──
        draw_grid_overlay(frame_show, pts_2d, visible, nx, ny, xs, ys, v)

        # fps
        t_now = time.perf_counter()
        dt = t_now - t_prev
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt
        t_prev = t_now
        cv2.putText(frame_show, f"{fps:.0f} fps", (w - 100, 20), FONT, 0.45, C_HUD, 1)

        cv2.imshow(WIN_MAIN, frame_show)

        # ── keys ──
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("s"), ord("S")):
            save_path = HERE / "config" / "saved.jsonc"
            out = {
                "intrinsics": {
                    "fx": v["fx"],
                    "fy": v["fy"],
                    "cx": v["cx"],
                    "cy": v["cy"],
                    "distortion": {
                        "k1": v["k1"],
                        "k2": v["k2"],
                        "p1": v["p1"],
                        "p2": v["p2"],
                        "k3": v["k3"],
                    },
                },
                "extrinsics": {
                    "tx": v["tx"],
                    "ty": v["ty"],
                    "tz": v["tz"],
                    "yaw": v["yaw"],
                    "pitch": v["pitch"],
                    "roll": v["roll"],
                },
            }
            save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"Saved → {save_path}")

        # ── window close check ──
        if cv2.getWindowProperty(WIN_MAIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
