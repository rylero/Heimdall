#!/usr/bin/env python3
"""
Heimdall camera calibration tool — GUI with two tabs.

Intrinsics tab
  Load images from a directory OR capture live.
  Detects checkerboard corners, runs calibrateCamera, displays results.

Extrinsics tab
  Two separate frame sets — use live camera capture or directory load for each:

  Ground frames  (board flat anywhere on floor)  →  pitch + tz
    Press [G] or the green button to capture.

  Yaw frames  (board at a measured robot-frame position)  →  yaw  (optional)
    Tick "Enable yaw", enter the board corner position, then press [Y].
    Board orientation: columns along robot +X (forward), rows along +Y (left).
    solvePnP is run with objpoints shifted into robot frame, so the resulting
    rotation R maps robot→camera;  camera Z in robot frame = R[2,:].

IMAGE REQUIREMENTS
  Capture raw, native (un-rotated) frames — the same frame the intrinsics describe.
  The pipeline's "rotation" enum (0/90/180/270) is applied downstream and un-rotated
  in projection; it must NOT be baked into the calibration. Use tools/capture.py on
  the Jetson to grab frames offline, then load here.

CONFIG TARGET
  Reads/writes the runtime configs in config/cameras/*.jsonc, matched by the "id"
  field (filenames are arbitrary — cam1.jsonc holds id 0). Values are updated in
  place, preserving comments. Only keys already present are written.
"""

import json
import re
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

CONFIG_DIR  = Path(__file__).parent.parent / "config" / "cameras"
CANVAS_W    = 720
CANVAS_H    = 440   # intrinsics tab
EXTR_CAM_H  = 280   # extrinsics live preview (shorter to leave room for controls)
THUMB_H     = 64
MIN_CAPTURES = 10


# ── OpenCV helpers ─────────────────────────────────────────────────────────────

def _make_objp(board: tuple[int, int], square_m: float) -> np.ndarray:
    cols, rows = board
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return objp


def _detect_corners(img: np.ndarray, board: tuple[int, int]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, board, flags)
    if ret:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return ret, corners


def _frame_to_tk(frame: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def _load_dir(directory: str) -> list[tuple[str, np.ndarray]]:
    p = Path(directory)
    files = sorted(f for ext in ("*.png", "*.jpg", "*.jpeg") for f in p.glob(ext))
    result = [(str(f), cv2.imread(str(f))) for f in files]
    return [(n, img) for n, img in result if img is not None]


def _strip_jsonc(text: str) -> str:
    """Drop // and /* */ comments so the runtime .jsonc configs parse with json.loads.
    (Matches nlohmann's ignore_comments in camera_config_loader.cpp. Assumes no '//'
    or '/*' appears inside a string value — true for these config files.)"""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _find_config(cam_id: int) -> tuple[Path, dict]:
    """Resolve the runtime camera config whose "id" == cam_id, across config/cameras/*.jsonc.
    The runtime loader (camera_config_loader.cpp) globs *.jsonc and keys on the id field —
    filenames are arbitrary (cam1.jsonc holds id 0), so match on id, not name."""
    for p in sorted(CONFIG_DIR.glob("*.jsonc")):
        try:
            cfg = json.loads(_strip_jsonc(p.read_text()))
        except Exception:
            continue
        if cfg.get("id") == cam_id:
            return p, cfg
    raise FileNotFoundError(
        f"no *.jsonc in {CONFIG_DIR} has \"id\": {cam_id} "
        f"(found: {[p.name for p in sorted(CONFIG_DIR.glob('*.jsonc'))]})")


def _update_jsonc_values(path: Path, updates: dict[str, float]) -> list[str]:
    """Surgically overwrite existing numeric values in a .jsonc file, preserving comments
    and formatting. Each key ("fx", "yaw", ...) is unique across these configs. Returns the
    list of keys that were NOT found (so the caller can warn instead of silently dropping)."""
    text = path.read_text()
    missing: list[str] = []
    for key, val in updates.items():
        pattern = re.compile(rf'("{re.escape(key)}"\s*:\s*)-?[\d.]+(?:[eE][+-]?\d+)?')
        text, n = pattern.subn(lambda m: m.group(1) + repr(float(val)), text, count=1)
        if n == 0:
            missing.append(key)
    path.write_text(text)
    return missing


def _load_existing_K(cam_id: int):
    _, cfg = _find_config(cam_id)
    intr = cfg["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    d = intr["distortion"]
    dist = np.array([[d["k1"], d["k2"], d["p1"], d["p2"], d["k3"]]], np.float64)
    return K, dist


# ── App ────────────────────────────────────────────────────────────────────────

class CalibApp:
    def __init__(self, root: tk.Tk, square_mm: float = 25.0,
                 cols: int = 9, rows: int = 6, cam_id: int = 0):
        self.root = root
        self.root.title("Heimdall Camera Calibration")

        self._build_top_bar()
        self._build_notebook()
        self._build_status()

        # apply CLI-provided board defaults
        self.sq_var.set(square_mm)
        self.cols_var.set(cols)
        self.rows_var.set(rows)
        self.cam_id_var.set(cam_id)

        # ── shared live camera ──────────────────────────────────────────────
        self.cap: cv2.VideoCapture | None = None
        self.live_frame: np.ndarray | None = None
        self._live_running = False

        # ── intrinsics state ────────────────────────────────────────────────
        self.objpoints: list[np.ndarray] = []
        self.imgpoints: list[np.ndarray] = []
        self.frame_size: tuple[int, int] | None = None
        self.intr_vis_frames: list[np.ndarray] = []   # corner-overlay images
        self.intr_K: np.ndarray | None = None
        self.intr_dist: np.ndarray | None = None

        # ── extrinsics state ─────────────────────────────────────────────────
        self.extr_ground_frames: list[np.ndarray] = []  # raw images → pitch + tz
        self.extr_yaw_frames: list[np.ndarray] = []     # raw images → yaw
        self.extr_result: dict[str, float] | None = None

        # ── key bindings ────────────────────────────────────────────────────
        self.root.bind("<space>", lambda _: self._intr_capture())
        self.root.bind("<g>",     lambda _: self._extr_capture_ground())
        self.root.bind("<y>",     lambda _: self._extr_capture_yaw())

    # ── top bar ─────────────────────────────────────────────────────────────

    def _build_top_bar(self):
        bar = tk.Frame(self.root, relief=tk.GROOVE, bd=1)
        bar.pack(fill=tk.X, padx=6, pady=4)

        tk.Label(bar, text="Camera ID:").pack(side=tk.LEFT)
        self.cam_id_var = tk.IntVar(value=0)
        tk.Spinbox(bar, from_=0, to=10, textvariable=self.cam_id_var, width=3
                   ).pack(side=tk.LEFT, padx=(2, 16))

        tk.Label(bar, text="Board:").pack(side=tk.LEFT)
        self.cols_var = tk.IntVar(value=9)
        self.rows_var = tk.IntVar(value=6)
        self.sq_var   = tk.DoubleVar(value=25.0)
        for lbl, var in (("cols", self.cols_var), ("rows", self.rows_var)):
            tk.Label(bar, text=lbl).pack(side=tk.LEFT)
            tk.Spinbox(bar, from_=3, to=20, textvariable=var, width=3).pack(side=tk.LEFT, padx=2)
        tk.Label(bar, text="size").pack(side=tk.LEFT)
        tk.Entry(bar, textvariable=self.sq_var, width=5).pack(side=tk.LEFT, padx=2)
        tk.Label(bar, text="mm").pack(side=tk.LEFT, padx=(0, 16))

        # shared camera open button
        tk.Button(bar, text="Open camera", command=self._open_live).pack(side=tk.LEFT)
        self.live_status_var = tk.StringVar(value="(no camera)")
        tk.Label(bar, textvariable=self.live_status_var, fg="#555").pack(side=tk.LEFT, padx=6)

    # ── notebook ────────────────────────────────────────────────────────────

    def _build_notebook(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)
        self._build_intr_tab()
        self._build_extr_tab()

    # ── intrinsics tab ───────────────────────────────────────────────────────

    def _build_intr_tab(self):
        f = tk.Frame(self.nb)
        self.nb.add(f, text="Intrinsics")

        # source row
        src = tk.LabelFrame(f, text="Source", padx=4, pady=3)
        src.pack(fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(src, text="Directory:").pack(side=tk.LEFT)
        self.intr_dir = tk.Entry(src, width=44)
        self.intr_dir.pack(side=tk.LEFT, padx=2)
        tk.Button(src, text="Browse", command=self._intr_browse).pack(side=tk.LEFT)
        tk.Button(src, text="Load", command=self._intr_load_dir,
                  bg="#2d7d2d", fg="white").pack(side=tk.LEFT, padx=4)

        # main area
        main = tk.Frame(f)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.intr_canvas = tk.Canvas(main, width=CANVAS_W, height=CANVAS_H, bg="#111")
        self.intr_canvas.pack(side=tk.LEFT)

        ctrl = tk.Frame(main, width=210)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        ctrl.pack_propagate(False)

        self.intr_count_var = tk.StringVar(value="Captures: 0")
        tk.Label(ctrl, textvariable=self.intr_count_var,
                 font=("monospace", 11)).pack(pady=4)

        tk.Button(ctrl, text="Capture  [Space]", font=("", 11, "bold"),
                  bg="#2d7d2d", fg="white", command=self._intr_capture
                  ).pack(fill=tk.X, pady=2)
        tk.Button(ctrl, text="Clear", command=self._intr_clear).pack(fill=tk.X, pady=2)

        ttk.Separator(ctrl).pack(fill=tk.X, pady=6)

        self.intr_cal_btn = tk.Button(ctrl, text="Calibrate", font=("", 11, "bold"),
                                       bg="#1a5faa", fg="white", state=tk.DISABLED,
                                       command=self._intr_calibrate)
        self.intr_cal_btn.pack(fill=tk.X, pady=2)

        self.intr_write_btn = tk.Button(ctrl, text="Write to JSON", bg="#884400", fg="white",
                                         state=tk.DISABLED, command=self._intr_write)
        self.intr_write_btn.pack(fill=tk.X, pady=2)

        ttk.Separator(ctrl).pack(fill=tk.X, pady=6)
        tk.Label(ctrl, text="Results:", anchor="w").pack(fill=tk.X)
        self.intr_result_box = scrolledtext.ScrolledText(ctrl, height=14, width=26,
                                                          font=("monospace", 8),
                                                          state=tk.DISABLED)
        self.intr_result_box.pack(fill=tk.BOTH, expand=True)

        # thumbnail strip
        strip_f = tk.Frame(f, height=THUMB_H + 4)
        strip_f.pack(fill=tk.X, padx=6)
        strip_f.pack_propagate(False)
        self.intr_strip = tk.Canvas(strip_f, height=THUMB_H + 4, bg="#222")
        self.intr_strip.pack(fill=tk.X)
        self.intr_thumbs: list[ImageTk.PhotoImage] = []

    # ── extrinsics tab ───────────────────────────────────────────────────────

    def _build_extr_tab(self):
        f = tk.Frame(self.nb)
        self.nb.add(f, text="Extrinsics  (ground board)")

        # live preview
        self.extr_canvas = tk.Canvas(f, width=CANVAS_W, height=EXTR_CAM_H, bg="#111")
        self.extr_canvas.pack(padx=6, pady=(4, 2))

        # ── ground frames ────────────────────────────────────────────────────
        gf = tk.LabelFrame(f,
                            text="Ground frames  —  board flat on floor, any position  →  pitch + tz",
                            padx=4, pady=3)
        gf.pack(fill=tk.X, padx=6, pady=2)

        tk.Button(gf, text="Capture [G]", bg="#2d7d2d", fg="white",
                  command=self._extr_capture_ground).pack(side=tk.LEFT)
        self.extr_gnd_count = tk.StringVar(value="0 frames")
        tk.Label(gf, textvariable=self.extr_gnd_count,
                 font=("monospace", 10), width=10).pack(side=tk.LEFT, padx=6)
        tk.Button(gf, text="Clear", command=self._extr_clear_ground).pack(side=tk.LEFT)
        ttk.Separator(gf, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)
        tk.Label(gf, text="or load dir:").pack(side=tk.LEFT)
        self.extr_gnd_dir = tk.Entry(gf, width=32)
        self.extr_gnd_dir.pack(side=tk.LEFT, padx=2)
        tk.Button(gf, text="Browse",
                  command=lambda: self._browse_into(self.extr_gnd_dir)).pack(side=tk.LEFT)
        tk.Button(gf, text="Load",
                  command=self._extr_load_ground_dir).pack(side=tk.LEFT, padx=2)

        # ── yaw frames ───────────────────────────────────────────────────────
        yf = tk.LabelFrame(f,
                            text="Yaw frames  —  board at measured robot-frame position  →  yaw (optional)",
                            padx=4, pady=3)
        yf.pack(fill=tk.X, padx=6, pady=2)

        self.extr_yaw_en = tk.BooleanVar(value=False)
        tk.Checkbutton(yf, text="Enable yaw calibration",
                       variable=self.extr_yaw_en).pack(anchor="w")

        pos_row = tk.Frame(yf)
        pos_row.pack(anchor="w")
        tk.Label(pos_row, text="Center of near board edge in robot frame:  X =").pack(side=tk.LEFT)
        self.extr_cx = tk.DoubleVar(value=1.0)
        tk.Entry(pos_row, textvariable=self.extr_cx, width=6).pack(side=tk.LEFT)
        tk.Label(pos_row, text=" m    Y =").pack(side=tk.LEFT)
        self.extr_cy = tk.DoubleVar(value=0.0)
        tk.Entry(pos_row, textvariable=self.extr_cy, width=6).pack(side=tk.LEFT)
        tk.Label(pos_row, text=" m    (+X forward, +Y left)").pack(side=tk.LEFT)
        tk.Label(yf, text="Near edge = row of corners closest to robot.  Align: columns → +Y (lateral)  rows → +X (forward).",
                 font=("", 8), fg="#666").pack(anchor="w")

        cap_row = tk.Frame(yf)
        cap_row.pack(anchor="w", pady=(2, 0))
        tk.Button(cap_row, text="Capture [Y]", bg="#5555aa", fg="white",
                  command=self._extr_capture_yaw).pack(side=tk.LEFT)
        self.extr_yaw_count = tk.StringVar(value="0 frames")
        tk.Label(cap_row, textvariable=self.extr_yaw_count,
                 font=("monospace", 10), width=10).pack(side=tk.LEFT, padx=6)
        tk.Button(cap_row, text="Clear", command=self._extr_clear_yaw).pack(side=tk.LEFT)
        ttk.Separator(cap_row, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)
        tk.Label(cap_row, text="or load dir:").pack(side=tk.LEFT)
        self.extr_yaw_dir = tk.Entry(cap_row, width=32)
        self.extr_yaw_dir.pack(side=tk.LEFT, padx=2)
        tk.Button(cap_row, text="Browse",
                  command=lambda: self._browse_into(self.extr_yaw_dir)).pack(side=tk.LEFT)
        tk.Button(cap_row, text="Load",
                  command=self._extr_load_yaw_dir).pack(side=tk.LEFT, padx=2)

        # ── run + results ─────────────────────────────────────────────────────
        bottom = tk.Frame(f)
        bottom.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        left = tk.Frame(bottom)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Button(left, text="Run Calibration", font=("", 11, "bold"),
                  bg="#1a5faa", fg="white", command=self._extr_run
                  ).pack(fill=tk.X, pady=(0, 4))
        tk.Label(left, text="Per-frame log:", anchor="w").pack(fill=tk.X)
        self.extr_log = scrolledtext.ScrolledText(left, height=6, font=("monospace", 9),
                                                   state=tk.DISABLED)
        self.extr_log.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(bottom, width=200)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)
        tk.Label(right, text="Summary:", font=("", 10, "bold")).pack(anchor="w", pady=4)
        self.extr_sum_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self.extr_sum_var, justify=tk.LEFT,
                 font=("monospace", 10)).pack(anchor="w")
        ttk.Separator(right).pack(fill=tk.X, pady=8)
        self.extr_write_btn = tk.Button(right, text="Write to JSON",
                                         bg="#884400", fg="white", font=("", 10, "bold"),
                                         state=tk.DISABLED, command=self._extr_write)
        self.extr_write_btn.pack(fill=tk.X)

    # ── status bar ──────────────────────────────────────────────────────────

    def _build_status(self):
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 relief=tk.SUNKEN, bg="#e8e8e8").pack(fill=tk.X, side=tk.BOTTOM)

    # ── board helpers ────────────────────────────────────────────────────────

    @property
    def _board(self) -> tuple[int, int]:
        return (self.cols_var.get(), self.rows_var.get())

    @property
    def _square_m(self) -> float:
        return self.sq_var.get() / 1000.0

    # ── shared live camera ───────────────────────────────────────────────────

    def _open_live(self):
        self._stop_live()
        idx = self.cam_id_var.get()
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {idx}")
            return
        self.cap = cap
        self._live_running = True
        threading.Thread(target=self._live_loop, daemon=True).start()
        self.root.after(33, self._live_poll)
        self.live_status_var.set(f"camera {idx} open")
        self.status_var.set(f"Camera {idx} open — Space=intrinsics capture  G=ground frame  Y=yaw frame")

    def _live_loop(self):
        while self._live_running and self.cap:
            ok, frame = self.cap.read()
            if ok:
                self.live_frame = frame

    def _live_poll(self):
        if self._live_running and self.live_frame is not None:
            frame = self.live_frame
            self._show_on(self.intr_canvas, frame, CANVAS_W, CANVAS_H)
            self._show_on(self.extr_canvas, frame, CANVAS_W, EXTR_CAM_H)
        if self._live_running:
            self.root.after(33, self._live_poll)

    def _stop_live(self):
        self._live_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.live_status_var.set("(no camera)")

    def _show_on(self, canvas: tk.Canvas, frame: np.ndarray, w: int, h: int):
        tk_img = _frame_to_tk(frame, w, h)
        canvas.create_image(w // 2, h // 2, anchor="center", image=tk_img)
        canvas.tk_img = tk_img

    # ── intrinsics: directory load ────────────────────────────────────────────

    def _intr_browse(self):
        d = filedialog.askdirectory(title="Select intrinsics image directory")
        if d:
            self.intr_dir.delete(0, tk.END)
            self.intr_dir.insert(0, d)

    def _intr_load_dir(self):
        d = self.intr_dir.get().strip()
        if not d:
            messagebox.showwarning("No path", "Enter or browse to an image directory.")
            return
        imgs = _load_dir(d)
        if not imgs:
            messagebox.showerror("Empty", f"No images in:\n{d}")
            return
        self._intr_clear()
        objp = _make_objp(self._board, self._square_m)
        ok = 0
        for fname, img in imgs:
            ret, corners = _detect_corners(img, self._board)
            if not ret:
                continue
            vis = img.copy()
            cv2.drawChessboardCorners(vis, self._board, corners, True)
            self.objpoints.append(objp.copy())
            self.imgpoints.append(corners)
            if self.frame_size is None:
                self.frame_size = img.shape[1], img.shape[0]
            self.intr_vis_frames.append(vis)
            ok += 1
        self.status_var.set(f"Loaded: {ok}/{len(imgs)} images with board detected.")
        self._intr_refresh()
        if self.intr_vis_frames:
            self._show_on(self.intr_canvas, self.intr_vis_frames[-1], CANVAS_W, CANVAS_H)

    # ── intrinsics: live capture ──────────────────────────────────────────────

    def _intr_capture(self):
        if not self._live_running or self.live_frame is None:
            return
        frame = self.live_frame.copy()
        objp = _make_objp(self._board, self._square_m)
        ret, corners = _detect_corners(frame, self._board)
        if not ret:
            self.status_var.set("Board not found — adjust angle/distance.")
            return
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, self._board, corners, True)
        self.objpoints.append(objp)
        self.imgpoints.append(corners)
        if self.frame_size is None:
            self.frame_size = frame.shape[1], frame.shape[0]
        self.intr_vis_frames.append(vis)
        self._intr_refresh()
        self._show_on(self.intr_canvas, vis, CANVAS_W, CANVAS_H)

    def _intr_clear(self):
        self.objpoints.clear()
        self.imgpoints.clear()
        self.intr_vis_frames.clear()
        self.frame_size = None
        self.intr_K = None
        self.intr_dist = None
        self._intr_refresh()
        self.intr_canvas.delete("all")
        self.intr_strip.delete("all")
        self.intr_thumbs.clear()
        self.intr_result_box.config(state=tk.NORMAL)
        self.intr_result_box.delete("1.0", tk.END)
        self.intr_result_box.config(state=tk.DISABLED)

    def _intr_refresh(self):
        n = len(self.intr_vis_frames)
        self.intr_count_var.set(f"Captures: {n}")
        self.intr_cal_btn.config(state=tk.NORMAL if n >= MIN_CAPTURES else tk.DISABLED)
        self._update_strip()

    def _update_strip(self):
        self.intr_strip.delete("all")
        self.intr_thumbs.clear()
        tw = int(THUMB_H * 4 / 3)
        x = 2
        for vis in self.intr_vis_frames[-18:]:
            thumb = _frame_to_tk(vis, tw, THUMB_H)
            self.intr_strip.create_image(x, 2, anchor="nw", image=thumb)
            self.intr_thumbs.append(thumb)
            x += tw + 2

    # ── intrinsics: calibrate + write ─────────────────────────────────────────

    def _intr_calibrate(self):
        if len(self.objpoints) < MIN_CAPTURES:
            return
        self.status_var.set("Calibrating…")
        self.root.update()
        rms, K, dist, *_ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.frame_size, None, None)
        self.intr_K, self.intr_dist = K, dist
        d = dist.flatten()
        self.intr_result_box.config(state=tk.NORMAL)
        self.intr_result_box.delete("1.0", tk.END)
        self.intr_result_box.insert(tk.END,
            f"RMS:  {rms:.4f} px\n\n"
            f"fx:   {K[0,0]:.4f}\nfy:   {K[1,1]:.4f}\n"
            f"cx:   {K[0,2]:.4f}\ncy:   {K[1,2]:.4f}\n\n"
            f"k1:   {d[0]:.6f}\nk2:   {d[1]:.6f}\n"
            f"p1:   {d[2]:.6f}\np2:   {d[3]:.6f}\nk3:   {d[4]:.6f}\n")
        self.intr_result_box.config(state=tk.DISABLED)
        self.intr_write_btn.config(state=tk.NORMAL)
        q = "excellent" if rms < 0.5 else "good" if rms < 1.0 else "poor — capture more varied poses"
        self.status_var.set(f"Intrinsics done. RMS={rms:.3f}px ({q})")

    def _intr_write(self):
        if self.intr_K is None:
            return
        try:
            cfg_path, _ = _find_config(self.cam_id_var.get())
        except FileNotFoundError as e:
            messagebox.showerror("Not found", str(e))
            return
        K, d = self.intr_K, self.intr_dist.flatten()
        updates = {
            "fx": round(float(K[0, 0]), 4), "fy": round(float(K[1, 1]), 4),
            "cx": round(float(K[0, 2]), 4), "cy": round(float(K[1, 2]), 4),
        }
        updates.update({k: round(float(v), 6)
                        for k, v in zip("k1 k2 p1 p2 k3".split(), d)})
        missing = _update_jsonc_values(cfg_path, updates)
        if missing:
            messagebox.showwarning("Missing keys",
                f"These keys were not present in {cfg_path.name} and were NOT written: "
                f"{', '.join(missing)}")
        messagebox.showinfo("Saved", f"Intrinsics → {cfg_path.name}")
        self.status_var.set(f"Intrinsics written → {cfg_path.name}")

    # ── extrinsics: live capture ──────────────────────────────────────────────

    def _extr_capture_ground(self):
        if not self._live_running or self.live_frame is None:
            self.status_var.set("Open camera first (top-bar button).")
            return
        frame = self.live_frame.copy()
        ret, corners = _detect_corners(frame, self._board)
        if not ret:
            self.status_var.set("Board not found in ground frame.")
            return
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, self._board, corners, True)
        self._show_on(self.extr_canvas, vis, CANVAS_W, EXTR_CAM_H)
        self.extr_ground_frames.append(frame)
        self.extr_gnd_count.set(f"{len(self.extr_ground_frames)} frames")
        self.status_var.set(f"Ground frame {len(self.extr_ground_frames)} captured.")

    def _extr_capture_yaw(self):
        if not self._live_running or self.live_frame is None:
            self.status_var.set("Open camera first (top-bar button).")
            return
        frame = self.live_frame.copy()
        ret, corners = _detect_corners(frame, self._board)
        if not ret:
            self.status_var.set("Board not found in yaw frame.")
            return
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, self._board, corners, True)
        self._show_on(self.extr_canvas, vis, CANVAS_W, EXTR_CAM_H)
        self.extr_yaw_frames.append(frame)
        self.extr_yaw_count.set(f"{len(self.extr_yaw_frames)} frames")
        self.status_var.set(f"Yaw frame {len(self.extr_yaw_frames)} captured.")

    def _extr_clear_ground(self):
        self.extr_ground_frames.clear()
        self.extr_gnd_count.set("0 frames")

    def _extr_clear_yaw(self):
        self.extr_yaw_frames.clear()
        self.extr_yaw_count.set("0 frames")

    # ── extrinsics: directory load ────────────────────────────────────────────

    def _browse_into(self, entry: tk.Entry):
        d = filedialog.askdirectory()
        if d:
            entry.delete(0, tk.END)
            entry.insert(0, d)

    def _extr_load_ground_dir(self):
        d = self.extr_gnd_dir.get().strip()
        if not d:
            messagebox.showwarning("No path", "Enter or browse to a directory.")
            return
        imgs = _load_dir(d)
        self.extr_ground_frames = [img for _, img in imgs]
        self.extr_gnd_count.set(f"{len(imgs)} frames")
        self.status_var.set(f"Loaded {len(imgs)} ground frames from {Path(d).name}/")

    def _extr_load_yaw_dir(self):
        d = self.extr_yaw_dir.get().strip()
        if not d:
            messagebox.showwarning("No path", "Enter or browse to a directory.")
            return
        imgs = _load_dir(d)
        self.extr_yaw_frames = [img for _, img in imgs]
        self.extr_yaw_count.set(f"{len(imgs)} frames")
        self.status_var.set(f"Loaded {len(imgs)} yaw frames from {Path(d).name}/")

    # ── extrinsics: run calibration ───────────────────────────────────────────

    def _extr_run(self):
        if not self.extr_ground_frames:
            messagebox.showwarning("No frames",
                                   "Capture or load ground frames first (board flat on floor).")
            return

        cam_id = self.cam_id_var.get()
        if self.intr_K is not None:
            K, dist = self.intr_K, self.intr_dist
        else:
            try:
                K, dist = _load_existing_K(cam_id)
            except Exception as e:
                messagebox.showerror("No intrinsics",
                                     f"Calibrate intrinsics first.\n{e}")
                return

        use_yaw = self.extr_yaw_en.get()
        objp_floor = _make_objp(self._board, self._square_m)

        objp_robot = None
        if use_yaw:
            if not self.extr_yaw_frames:
                messagebox.showwarning("No yaw frames",
                                       "Capture or load yaw frames, or disable yaw calibration.")
                return
            # User specifies center of the near board edge (j=0 row, closest to robot).
            # Board orientation: columns along robot +Y (lateral), rows along robot +X (forward).
            # objp x-axis = columns → robot Y,  objp y-axis = rows → robot X.
            # Near-edge center in board local coords: (cols-1)/2 * sq along X, 0 along Y.
            # Corner (0,0) in robot frame:
            #   robot_X = cx_robot - 0                           (near edge, no row offset)
            #   robot_Y = cy_robot - (cols-1)/2 * square_m      (center to left edge)
            cx, cy = self.extr_cx.get(), self.extr_cy.get()
            half_width = (self.cols_var.get() - 1) / 2.0 * self._square_m
            # Shift objpoints so near-edge center = (cx, cy) in robot frame.
            # objp[:, 0] spans 0..cols-1 (lateral Y), objp[:, 1] spans 0..rows-1 (forward X).
            objp_robot = np.zeros_like(objp_floor)
            objp_robot[:, 0] = cy - half_width + objp_floor[:, 0]  # robot Y
            objp_robot[:, 1] = cx + objp_floor[:, 1]               # robot X (rows go forward)
            objp_robot[:, 2] = 0.0

        log_lines: list[str] = []
        pitches: list[float] = []
        heights: list[float] = []
        yaws: list[float] = []

        def _solve_ground(img: np.ndarray, label: str):
            ret, corners = _detect_corners(img, self._board)
            if not ret:
                log_lines.append(f"SKIP (no board)  {label}")
                return
            ok, rvec, tvec = cv2.solvePnP(objp_floor, corners, K, dist)
            if not ok:
                log_lines.append(f"FAIL (solvePnP)  {label}")
                return
            R, _ = cv2.Rodrigues(rvec)
            height = float((-R.T @ tvec).flatten()[2])
            cam_fwd = R[2, :]
            horiz = float(np.sqrt(cam_fwd[0] ** 2 + cam_fwd[1] ** 2))
            pitch = float(np.arctan2(-cam_fwd[2], horiz))
            warn = ""
            if not (0.05 < height < 2.0):
                warn += " ⚠tz"
            if not (0.0 < pitch < 1.6):
                warn += " ⚠pitch"
            log_lines.append(f"GND  {label:30s}  tz={height:.4f}m  pitch={np.degrees(pitch):.1f}°{warn}")
            pitches.append(pitch)
            heights.append(height)

        def _solve_yaw(img: np.ndarray, label: str):
            ret, corners = _detect_corners(img, self._board)
            if not ret:
                log_lines.append(f"SKIP (no board)  {label}")
                return
            ok, rvec, _ = cv2.solvePnP(objp_robot, corners, K, dist)
            if not ok:
                log_lines.append(f"FAIL (solvePnP)  {label}")
                return
            R, _ = cv2.Rodrigues(rvec)
            # R: robot→camera.  Camera Z axis in robot frame = R[2,:]
            cam_z_robot = R[2, :]
            yaw = float(np.arctan2(cam_z_robot[1], cam_z_robot[0]))
            log_lines.append(f"YAW  {label:30s}  yaw={np.degrees(yaw):.1f}°")
            yaws.append(yaw)

        for i, img in enumerate(self.extr_ground_frames):
            _solve_ground(img, f"ground_{i:03d}")
        if use_yaw:
            for i, img in enumerate(self.extr_yaw_frames):
                _solve_yaw(img, f"yaw_{i:03d}")

        self.extr_log.config(state=tk.NORMAL)
        self.extr_log.delete("1.0", tk.END)
        self.extr_log.insert(tk.END, "\n".join(log_lines))
        self.extr_log.config(state=tk.DISABLED)

        if not pitches:
            self.status_var.set("No valid ground frames.")
            return

        self.extr_result = {
            "pitch": float(np.mean(pitches)),
            "tz":    float(np.mean(heights)),
        }
        if yaws:
            self.extr_result["yaw"] = float(np.mean(yaws))

        summary = (
            f"pitch:  {np.degrees(np.mean(pitches)):.2f}°"
            f"  ±{np.degrees(np.std(pitches)):.2f}°\n"
            f"  ({self.extr_result['pitch']:.5f} rad)\n\n"
            f"tz:  {np.mean(heights):.4f} m"
            f"  ±{np.std(heights):.4f} m"
        )
        if yaws:
            summary += (
                f"\n\nyaw:  {np.degrees(np.mean(yaws)):.2f}°"
                f"  ±{np.degrees(np.std(yaws)):.2f}°\n"
                f"  ({self.extr_result['yaw']:.5f} rad)"
            )
        self.extr_sum_var.set(summary)
        self.extr_write_btn.config(state=tk.NORMAL)
        self.status_var.set(
            f"Extrinsics done — {len(pitches)} ground frames"
            + (f", {len(yaws)} yaw frames" if yaws else "")
        )

    # ── extrinsics: write ─────────────────────────────────────────────────────

    def _extr_write(self):
        if not self.extr_result:
            return
        try:
            cfg_path, _ = _find_config(self.cam_id_var.get())
        except FileNotFoundError as e:
            messagebox.showerror("Not found", str(e))
            return
        updates = {key: round(val, 6) for key, val in self.extr_result.items()}
        missing = _update_jsonc_values(cfg_path, updates)
        if missing:
            messagebox.showwarning("Missing keys",
                f"These keys were not present in {cfg_path.name} and were NOT written: "
                f"{', '.join(missing)}. Add them to extrinsics and re-run.")
        keys = ", ".join(self.extr_result)
        messagebox.showinfo("Saved", f"Extrinsics ({keys}) → {cfg_path.name}")
        self.status_var.set(f"Extrinsics written → {cfg_path.name}")

    # ── cleanup ──────────────────────────────────────────────────────────────

    def on_close(self):
        self._stop_live()
        self.root.destroy()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Heimdall camera calibration GUI")
    ap.add_argument("--square-mm", type=float, default=25.0, help="checkerboard square size in mm")
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--cam", type=int, default=0, help="camera id (matches config \"id\")")
    args = ap.parse_args()

    root = tk.Tk()
    app = CalibApp(root, square_mm=args.square_mm, cols=args.cols,
                   rows=args.rows, cam_id=args.cam)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
