"""
Camera calibration tool.

Point at a checkerboard, click Capture for 15-20 poses, then Calibrate.
Outputs intrinsics in Heimdall camera JSON format.

Dependencies: pip install opencv-python pillow
Usage:        python tools/calibrate.py [device_index]
"""

import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import json

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DEVICE  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PREVIEW_W       = 800
PREVIEW_H       = 600
MIN_CAPTURES    = 10   # minimum before Calibrate button enables


class CalibApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Heimdall Camera Calibration")
        self.root.resizable(False, False)

        # State
        self.cap          = None
        self.obj_points   = []   # 3-D checkerboard corners
        self.img_points   = []   # 2-D detected corners
        self.frame_size   = None
        self.last_frame   = None
        self.overlay      = None  # frame with corners drawn
        self.show_overlay = False

        self._build_ui()
        self._open_camera(DEFAULT_DEVICE)
        self._poll()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, padx=8, pady=8)

        self.canvas = tk.Canvas(left, width=PREVIEW_W, height=PREVIEW_H, bg="black")
        self.canvas.pack()

        status_row = tk.Frame(left)
        status_row.pack(fill=tk.X, pady=(4, 0))
        self.status_var = tk.StringVar(value="Open a camera to begin.")
        tk.Label(status_row, textvariable=self.status_var, anchor="w",
                 font=("monospace", 10)).pack(fill=tk.X)

        right = tk.Frame(self.root, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        right.pack_propagate(False)

        # Camera
        tk.Label(right, text="Camera device index:").pack(anchor="w")
        cam_row = tk.Frame(right); cam_row.pack(fill=tk.X, pady=(0, 8))
        self.cam_var = tk.IntVar(value=DEFAULT_DEVICE)
        tk.Spinbox(cam_row, from_=0, to=10, textvariable=self.cam_var,
                   width=4).pack(side=tk.LEFT)
        tk.Button(cam_row, text="Open",
                  command=lambda: self._open_camera(self.cam_var.get())
                  ).pack(side=tk.LEFT, padx=4)

        # Checkerboard size
        tk.Label(right, text="Checkerboard inner corners:").pack(anchor="w")
        grid_row = tk.Frame(right); grid_row.pack(fill=tk.X, pady=(0, 8))
        self.cols_var = tk.IntVar(value=9)
        self.rows_var = tk.IntVar(value=6)
        tk.Label(grid_row, text="cols").pack(side=tk.LEFT)
        tk.Spinbox(grid_row, from_=3, to=20, textvariable=self.cols_var,
                   width=4).pack(side=tk.LEFT, padx=2)
        tk.Label(grid_row, text="rows").pack(side=tk.LEFT)
        tk.Spinbox(grid_row, from_=3, to=20, textvariable=self.rows_var,
                   width=4).pack(side=tk.LEFT, padx=2)

        # Square size
        tk.Label(right, text="Square size (mm):").pack(anchor="w")
        self.sq_var = tk.DoubleVar(value=25.0)
        tk.Entry(right, textvariable=self.sq_var, width=8).pack(anchor="w",
                                                                  pady=(0, 8))

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=4)

        # Capture button
        self.capture_btn = tk.Button(right, text="Capture Frame  [Space]",
                                     font=("", 12, "bold"),
                                     bg="#2d7d2d", fg="white",
                                     state=tk.DISABLED,
                                     command=self._capture)
        self.capture_btn.pack(fill=tk.X, pady=4)

        self.count_var = tk.StringVar(value="Captured: 0")
        tk.Label(right, textvariable=self.count_var,
                 font=("monospace", 11)).pack(pady=2)

        tk.Button(right, text="Clear Captures",
                  command=self._clear).pack(fill=tk.X, pady=2)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=4)

        self.cal_btn = tk.Button(right, text="Calibrate",
                                 font=("", 12, "bold"),
                                 bg="#1a5faa", fg="white",
                                 state=tk.DISABLED,
                                 command=self._calibrate)
        self.cal_btn.pack(fill=tk.X, pady=4)

        tk.Label(right, text="Results (copy into camera JSON):").pack(anchor="w")
        self.result_box = scrolledtext.ScrolledText(right, height=16,
                                                    font=("monospace", 9),
                                                    state=tk.DISABLED)
        self.result_box.pack(fill=tk.BOTH, expand=True, pady=4)

        self.root.bind("<space>", lambda _: self._capture())

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _open_camera(self, idx: int):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open device {idx}")
            self.cap = None
            return
        self.capture_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Device {idx} opened. Show checkerboard and Capture.")

    # ------------------------------------------------------------------
    # Preview loop
    # ------------------------------------------------------------------

    def _poll(self):
        if self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                self.last_frame = frame
                display = self.overlay if self.show_overlay and self.overlay is not None \
                          else frame
                self._show(display)
                self.show_overlay = False   # flash overlay for one tick then return to live

        self.root.after(30, self._poll)

    def _show(self, frame):
        h, w = frame.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(PREVIEW_W // 2, PREVIEW_H // 2,
                                 anchor="center", image=img)
        self.canvas.image = img   # prevent GC

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture(self):
        if self.last_frame is None:
            return

        cols = self.cols_var.get()
        rows = self.rows_var.get()
        pattern = (cols, rows)
        gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not found:
            self.status_var.set("No checkerboard found — try a different angle.")
            return

        # Subpixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        sq_m = self.sq_var.get() / 1000.0   # mm → metres
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_m

        self.obj_points.append(objp)
        self.img_points.append(corners2)
        if self.frame_size is None:
            self.frame_size = (gray.shape[1], gray.shape[0])

        # Draw corners and flash on screen
        vis = self.last_frame.copy()
        cv2.drawChessboardCorners(vis, pattern, corners2, found)
        self.overlay = vis
        self.show_overlay = True

        n = len(self.obj_points)
        self.count_var.set(f"Captured: {n}")
        self.status_var.set(f"Frame {n} captured. {'Calibrate when ready.' if n >= MIN_CAPTURES else f'Need {MIN_CAPTURES - n} more.'}")

        if n >= MIN_CAPTURES:
            self.cal_btn.config(state=tk.NORMAL)

    def _clear(self):
        self.obj_points.clear()
        self.img_points.clear()
        self.frame_size = None
        self.count_var.set("Captured: 0")
        self.cal_btn.config(state=tk.DISABLED)
        self.status_var.set("Cleared. Capture new frames.")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate(self):
        if len(self.obj_points) < MIN_CAPTURES:
            return

        self.status_var.set("Calibrating…")
        self.root.update()

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, self.frame_size, None, None)

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # Per-image reprojection error
        errors = []
        for i, (op, ip) in enumerate(zip(self.obj_points, self.img_points)):
            proj, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], K, dist)
            errors.append(float(np.sqrt(np.mean((ip - proj) ** 2))))
        mean_err = float(np.mean(errors))

        intrinsics = {"fx": round(fx, 4), "fy": round(fy, 4),
                      "cx": round(cx, 4), "cy": round(cy, 4)}

        dist_list = dist.flatten().tolist()

        output = {
            "intrinsics": intrinsics,
            "_reprojection_error_px": round(mean_err, 4),
            "_distortion_k1_k2_p1_p2_k3": [round(v, 6) for v in dist_list],
            "_resolution": list(self.frame_size),
            "_num_frames": len(self.obj_points),
        }

        result_str = json.dumps(output, indent=2)

        self.result_box.config(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, result_str)
        self.result_box.config(state=tk.DISABLED)

        quality = "excellent" if mean_err < 0.5 else "good" if mean_err < 1.0 else "poor — capture more varied poses"
        self.status_var.set(
            f"Done. RMS reprojection error: {mean_err:.3f} px ({quality})")

        print("\n=== Calibration Result ===")
        print(result_str)

    # ------------------------------------------------------------------

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CalibApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
