"""
projection.py -- the single source of truth for the sim camera model.

This module is the ONLY place the Python side models the camera geometry. It
mirrors the C++ pipeline exactly:
  - rotation_from_euler()  matches rotation_from_euler() in camera_params.h
  - distort()               is the exact forward of pose_estimator.cpp's iterative
                            Brown-Conrady undistortion (k1,k2,p1,p2,k3)
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
            'width':        640, 'height': 480, 'rotation': 0,
            'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0,
            'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0,  # ideal pinhole
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

        # Intrinsics stay native (calibrated on the raw feed). The pipeline rotates the frame
        # before nvinfer, so the sim forward-projects to native pixels then rotates the bbox
        # into the nvinfer frame (see field_to_pixel); the C++ un-rotates it back. This mirrors
        # camera_config_loader.cpp, which no longer fudges intrinsics for flips.
        rotation = int(cfg.get('rotation', 0))
        if rotation not in (0, 90, 180, 270):
            raise RuntimeError(f"{path}: rotation must be 0/90/180/270 (got {rotation})")

        # Brown-Conrady distortion coefficients (the C++ pose_estimator undistorts
        # with these, so the sim must forward-distort with them for the round-trip
        # to close). Absent -> zero (ideal pinhole).
        dist = intr.get('distortion', {})
        R = rotation_from_euler(float(extr['yaw']), float(extr['pitch']), float(extr['roll']))
        cameras.append({
            'id': cfg['id'],
            'width': w, 'height': h, 'rotation': rotation,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'k1': float(dist.get('k1', 0.0)), 'k2': float(dist.get('k2', 0.0)),
            'p1': float(dist.get('p1', 0.0)), 'p2': float(dist.get('p2', 0.0)),
            'k3': float(dist.get('k3', 0.0)),
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


# ── Lens distortion (forward Brown-Conrady) ──────────────────────────────────
# This is the exact inverse of the undistortion iteration in pose_estimator.cpp:
#   xd = u*rad + 2*p1*u*v + p2*(r2 + 2u^2)
#   yd = v*rad + p1*(r2 + 2v^2) + 2*p2*u*v,   rad = 1 + k1 r2 + k2 r4 + k3 r6
# The sim projects field->pixel, so it must forward-distort; the C++ then
# undistorts, and the round-trip closes. Without this the C++ removes distortion
# that was never added, which is the source of the systematic pose offset.

def distort(cam, u, v):
    """Apply forward Brown-Conrady distortion to normalised coords (u, v)."""
    k1, k2, k3 = cam['k1'], cam['k2'], cam['k3']
    p1, p2 = cam['p1'], cam['p2']
    r2 = u*u + v*v
    rad = 1.0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xd = u*rad + 2.0*p1*u*v + p2*(r2 + 2.0*u*u)
    yd = v*rad + p1*(r2 + 2.0*v*v) + 2.0*p2*u*v
    return xd, yd


# ── Projection: field point -> pixel bbox ─────────────────────────────────────

def field_to_pixel(cam, robot_x, robot_y, robot_heading, obj_x, obj_y, obj_radius=0.12):
    """
    Projects a field-space circle through the camera model -> pixel bbox.
    Returns (left, top, width, height) or None if not visible.

    Applies forward lens distortion so the round-trip through the C++
    pose_estimator (which undistorts) recovers the true field point.
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
    # Normalised undistorted coords -> distorted -> pixels (matches C++ inverse).
    xd, yd = distort(cam, d_cam[0] / d_cam[2], d_cam[1] / d_cam[2])
    u = fx * xd + cx
    v = fy * yd + cy

    # Pixel radius from angular subtense of object radius.
    # pose_estimator.cpp reads ground contact as (left+w/2, top+h) -- the BOTTOM-CENTER.
    # So u,v (the ground projection) must land at top+h, meaning top = v - h.
    px_r = abs(fx) * obj_radius / d_cam[2]

    w    = 2.0 * px_r;  h = 2.0 * px_r
    left = u - px_r
    top  = v - h  # bottom-center = (u, v) = ground contact projection

    if left + w < 0 or left > cam['width'] or top + h < 0 or top > cam['height']:
        return None

    # Emit the bbox in the nvinfer (rotated) frame, matching what the real pipeline feeds the
    # detector. The C++ pose estimator un-rotates it back (ground_contact_native), closing the
    # round-trip. This is the forward of that inverse map.
    return _rotate_bbox_native_to_frame(cam, left, top, w, h)


def _rotate_bbox_native_to_frame(cam, left, top, w, h):
    """Rotate a native-frame bbox into the pipeline's rotated (nvinfer) frame.
    Forward of pose_estimator.cpp::ground_contact_native's inverse map; keeps boxes
    axis-aligned for 0/90/180/270."""
    rot = cam.get('rotation', 0)
    if rot == 0:
        return (left, top, w, h)
    W = float(cam['width']);  H = float(cam['height'])
    corners = [(left, top), (left + w, top), (left, top + h), (left + w, top + h)]
    pts = []
    for xn, yn in corners:
        if rot == 90:     xr, yr = (H - 1.0) - yn, xn            # native -> cw90 frame
        elif rot == 180:  xr, yr = (W - 1.0) - xn, (H - 1.0) - yn
        else:             xr, yr = yn, (W - 1.0) - xn            # 270
        pts.append((xr, yr))
    xs = [p[0] for p in pts];  ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
