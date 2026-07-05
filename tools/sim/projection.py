"""
projection.py -- the single source of truth for the sim camera model.

This module is the ONLY place the Python side models the camera geometry. It
mirrors the C++ pipeline exactly:
  - rotation_from_euler()  matches rotation_from_euler() in camera_params.h
  - the fudge factor        matches pose_estimator.cpp (kA, kB, kC)
  - field_to_pixel()        uses the bottom-center ground-contact convention
                            that pose_estimator.cpp reads back (left+w/2, top+h)

sim_recording.py and compare_replay.py both import from here so the model lives
in one place. If this drifts from the C++, the sim lies -- guard it with the
round-trip test in tests/ (project -> heimdall_replay -> recovered field pos).
"""
import json
import math
import os
import re


# ── Linear algebra helpers ─────────────────────────────────────────────────────

def mat3_mul(A, B):
    C = [0.0] * 9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i*3+j] += A[i*3+k] * B[k*3+j]
    return C

def mat3_transpose(M):
    return [M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8]]

def mat3_vec(M, v):
    return (
        M[0]*v[0] + M[1]*v[1] + M[2]*v[2],
        M[3]*v[0] + M[4]*v[1] + M[5]*v[2],
        M[6]*v[0] + M[7]*v[1] + M[8]*v[2],
    )

def rotation_from_euler(yaw, pitch, roll):
    """Matches rotation_from_euler() in camera_params.h exactly."""
    R_base = [0, 0, 1, -1, 0, 0, 0, -1, 0]
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    Rx = [1, 0, 0, 0, cp, -sp, 0, sp, cp]
    cr, sr = math.cos(roll), math.sin(roll)
    Rz_cam = [cr, -sr, 0, sr, cr, 0, 0, 0, 1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz_rob = [cy, -sy, 0, sy, cy, 0, 0, 0, 1]
    return mat3_mul(Rz_rob, mat3_mul(mat3_mul(R_base, Rx), Rz_cam))


# ── Camera utilities ───────────────────────────────────────────────────────────

def camera_sees_floor(cam):
    """True if the camera's optical axis (cam Z) has a downward component in robot frame.
    R_cam_to_rob applied to [0,0,1] gives cam Z in robot frame.
    R_cam_to_rob = R_rob_to_cam^T, so cam Z in robot frame = column 2 of R_cam_to_rob
                 = row 2 of R_rob_to_cam = (Rt[6], Rt[7], Rt[8]).
    Ground-facing requires this z-component < 0 (pointing down)."""
    Rt = cam['R_rob_to_cam']
    cam_z_rob_z = Rt[6]*0 + Rt[7]*0 + Rt[8]*1  # = Rt[8]
    return cam_z_rob_z < -0.1

def make_sim_cameras():
    """Built-in cameras for simulation when config cameras can't see the floor.
    Two floor-facing cameras (pitch=+45 deg) on left and right of robot front."""
    cameras = []
    for i, (tx, ty) in enumerate([(0.20, 0.15), (0.20, -0.15)]):
        pitch = math.radians(45.0)  # 45° gives ~0.9m forward sight range at 0.30m height
        R = rotation_from_euler(0.0, pitch, 0.0)
        cameras.append({
            'id':           i,
            'width':        640, 'height': 480,
            'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0,
            'tx': tx, 'ty': ty, 'tz': 0.30,
            'R_rob_to_cam': mat3_transpose(R),
        })
    return cameras


# ── Camera config loading ──────────────────────────────────────────────────────

def strip_comments(text):
    return re.sub(r'//[^\n]*', '', text)

def load_cameras(cam_dir):
    """Load camera configs the same way the live system does (.jsonc, // comments)."""
    paths = sorted(
        os.path.join(cam_dir, f)
        for f in os.listdir(cam_dir)
        if f.endswith('.jsonc')
    )
    if not paths:
        raise RuntimeError(f"No .jsonc files in {cam_dir}")

    cameras = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            cfg = json.loads(strip_comments(f.read()))

        intr = cfg['intrinsics']
        extr = cfg['extrinsics']
        fx = float(intr['fx']);  fy = float(intr['fy'])
        cx = float(intr['cx']);  cy = float(intr['cy'])
        w  = cfg.get('width', 640);  h = cfg.get('height', 480)

        if cfg.get('flip_h', False):
            cx = w - 1.0 - cx;  fx = -fx
        if cfg.get('flip_v', False):
            cy = h - 1.0 - cy

        R = rotation_from_euler(float(extr['yaw']), float(extr['pitch']), float(extr['roll']))
        cameras.append({
            'id': cfg['id'],
            'width': w, 'height': h,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'tx': float(extr['tx']), 'ty': float(extr['ty']), 'tz': float(extr['tz']),
            'R_rob_to_cam': mat3_transpose(R),  # R^T = robot-to-camera
        })
    return cameras

def load_cameras_or_sim(cam_dir):
    """Load configured cameras; fall back to built-in sim cameras if none can see
    the floor. Returns (cameras, used_fallback)."""
    cameras = load_cameras(cam_dir)
    if not any(camera_sees_floor(c) for c in cameras):
        return make_sim_cameras(), True
    return cameras, False


# ── Fudge factor (mirrors pose_estimator.cpp) ────────────────────────────────
# pose_estimator applies: corrected_pos = raw_pos * (corrected_d / raw_d)
# where corrected_d = max(0, kA*raw_d^2 + kB*raw_d + kC).
_kA, _kB, _kC = -0.0658, 0.9637, 0.12

def apply_fudge(fx, fy):
    """Forward fudge: same transform as pose_estimator.cpp."""
    raw_d = math.hypot(fx, fy)
    if raw_d < 0.01:
        return fx, fy
    corr_d = max(0.0, (_kA * raw_d + _kB) * raw_d + _kC)
    s = corr_d / raw_d
    return fx * s, fy * s

def inv_fudge(gx, gy):
    """Inverse fudge: find raw position such that apply_fudge(raw) = (gx, gy)."""
    d_gt = math.hypot(gx, gy)
    if d_gt < 0.01:
        return gx, gy
    disc = _kB * _kB - 4 * _kA * (_kC - d_gt)
    if disc < 0:
        return gx, gy
    sq = math.sqrt(disc)
    r1 = (-_kB + sq) / (2 * _kA)
    r2 = (-_kB - sq) / (2 * _kA)
    raw_d = r2 if r2 > 0 and (r1 <= 0 or abs(r2 - d_gt) < abs(r1 - d_gt)) else r1
    if raw_d <= 0:
        return gx, gy
    s = raw_d / d_gt
    return gx * s, gy * s


# ── Projection: field point -> pixel bbox ─────────────────────────────────────

def field_to_pixel(cam, robot_x, robot_y, robot_heading, obj_x, obj_y, obj_radius=0.12):
    """
    Projects a field-space circle through the camera model -> pixel bbox.
    Returns (left, top, width, height) or None if not visible.

    NOTE: We do NOT apply inv_fudge here. The C++ fudge is calibrated at close
    range from field origin, so applying it to objects far down the field pushes
    their pre-compensated positions outside the camera's reachable FOV. Accept a
    small systematic offset vs ground truth instead.
    """
    # Field -> robot frame  (R(-heading) * delta)
    cos_h = math.cos(robot_heading);  sin_h = math.sin(robot_heading)
    dx = obj_x - robot_x;  dy = obj_y - robot_y
    rx =  cos_h * dx + sin_h * dy
    ry = -sin_h * dx + cos_h * dy
    rz = 0.0  # object is on the floor

    # Direction from camera origin to object, in robot frame
    d_rob = (rx - cam['tx'], ry - cam['ty'], rz - cam['tz'])

    # Rotate to camera frame
    d_cam = mat3_vec(cam['R_rob_to_cam'], d_rob)

    # Must be in front of camera (cam Z points toward scene)
    if d_cam[2] <= 0.05:
        return None

    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    u = fx * (d_cam[0] / d_cam[2]) + cx
    v = fy * (d_cam[1] / d_cam[2]) + cy

    # Pixel radius from angular subtense of object radius.
    # pose_estimator.cpp reads ground contact as (left+w/2, top+h) -- the BOTTOM-CENTER.
    # So u,v (the ground projection) must land at top+h, meaning top = v - h.
    px_r = abs(fx) * obj_radius / d_cam[2]

    w    = 2.0 * px_r;  h = 2.0 * px_r
    left = u - px_r
    top  = v - h  # bottom-center = (u, v) = ground contact projection

    if left + w < 0 or left > cam['width'] or top + h < 0 or top > cam['height']:
        return None

    return (left, top, w, h)
