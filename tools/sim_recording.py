"""
sim_recording.py -- Synthetic record/replay dataset generator

Simulates moving objects in field space, projects them to noisy pixel detections,
and writes two files:
  <out>_recording.jsonl     -- noisy detections + poses  (feed to heimdall_replay)
  <out>_ground_truth.jsonl  -- perfect field positions    (diff target)

Usage:
  python tools/sim_recording.py [--cameras config/cameras] [--out recordings/sim] [--duration 10]

Camera config dir is read exactly like the live system (.jsonc with // comments).
If the configured cameras cannot see the floor (e.g. placeholder pitch sign is wrong),
the sim falls back to built-in simulation cameras and prints a warning.
"""
import argparse
import json
import math
import os
import random
import re
import time


# ── Linear algebra helpers ─────────────────────────────────────────────────────

def mat3_mul(A, B):
    C = [0.0] * 9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i*3+j] += A[i*3+k] * B[k*3+j]
    return C

def mat3_transpose(M):
    return [M[0],M[3],M[6], M[1],M[4],M[7], M[2],M[5],M[8]]

def mat3_vec(M, v):
    return (
        M[0]*v[0] + M[1]*v[1] + M[2]*v[2],
        M[3]*v[0] + M[4]*v[1] + M[5]*v[2],
        M[6]*v[0] + M[7]*v[1] + M[8]*v[2],
    )

def rotation_from_euler(yaw, pitch, roll):
    """Matches rotation_from_euler() in camera_params.h exactly."""
    R_base = [0,0,1, -1,0,0, 0,-1,0]
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    Rx = [1,0,0, 0,cp,-sp, 0,sp,cp]
    cr, sr = math.cos(roll), math.sin(roll)
    Rz_cam = [cr,-sr,0, sr,cr,0, 0,0,1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz_rob = [cy,-sy,0, sy,cy,0, 0,0,1]
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
    Two floor-facing cameras (pitch=+70 deg) on left and right of robot front."""
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


# ── Fudge factor (mirrors pose_estimator.cpp) ────────────────────────────────
# pose_estimator applies: corrected_pos = raw_pos * (corrected_d / raw_d)
# where corrected_d = max(0, kA*raw_d^2 + kB*raw_d + kC).
# To pre-compensate in the sim, we project from inv_fudge(gt_pos) so that the
# C++ pipeline's forward fudge lands exactly on ground truth.
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
    # Solve kA*r^2 + kB*r + (kC - d_gt) = 0  for r = raw_d
    disc = _kB * _kB - 4 * _kA * (_kC - d_gt)
    if disc < 0:
        return gx, gy
    sq = math.sqrt(disc)
    # Two roots; pick the positive one closer to d_gt
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
    their pre-compensated positions outside the camera's reachable FOV. Accept
    a small systematic offset vs ground truth instead.
    """

    # Field -> robot frame  (R(-heading) * delta)
    cos_h = math.cos(robot_heading);  sin_h = math.sin(robot_heading)
    dx = obj_x - robot_x;  dy = obj_y - robot_y
    rx =  cos_h * dx + sin_h * dy
    ry = -sin_h * dx + cos_h * dy
    rz = 0.0  # object is on the floor

    # Direction from camera origin to object, in robot frame
    d_rob = (rx - cam['tx'], ry - cam['ty'], rz - cam['tz'])  # z = -tz (below camera)

    # Rotate to camera frame
    d_cam = mat3_vec(cam['R_rob_to_cam'], d_rob)

    # Must be in front of camera (cam Z points toward scene)
    if d_cam[2] <= 0.05:
        return None

    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    u = fx * (d_cam[0] / d_cam[2]) + cx
    v = fy * (d_cam[1] / d_cam[2]) + cy

    # Pixel radius from angular subtense of object radius.
    # pose_estimator.cpp reads ground contact as (left+w/2, top+h) — the BOTTOM-CENTER.
    # So u,v (the ground projection) must land at top+h, meaning top = v - h.
    px_r = abs(fx) * obj_radius / d_cam[2]

    w    = 2.0 * px_r;  h = 2.0 * px_r
    left = u - px_r
    top  = v - h  # bottom-center = (u, v) = ground contact projection

    if left + w < 0 or left > cam['width'] or top + h < 0 or top > cam['height']:
        return None

    return (left, top, w, h)


# ── Simulated objects ──────────────────────────────────────────────────────────

class SimObject:
    def __init__(self, obj_id, class_id, x0, y0, vx=0.0, vy=0.0):
        self.obj_id = obj_id;  self.class_id = class_id
        self.x0 = x0;  self.y0 = y0;  self.vx = vx;  self.vy = vy

    def position(self, t):
        return self.x0 + self.vx * t, self.y0 + self.vy * t


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic recording for heimdall_replay")
    ap.add_argument('--cameras',  default='config/cameras',  help='Camera config dir (.jsonc files)')
    ap.add_argument('--out',      default='recordings/sim',   help='Output path prefix (no extension)')
    ap.add_argument('--duration', type=float, default=10.0,   help='Simulation duration (seconds)')
    ap.add_argument('--fps',      type=float, default=30.0,   help='Detection frame rate')
    ap.add_argument('--pose-hz',  type=float, default=50.0,   help='Pose publish rate')
    ap.add_argument('--noise',    type=float, default=5.0,    help='Pixel noise std-dev (px)')
    ap.add_argument('--conf',     type=float, default=0.85,   help='Detection confidence')
    ap.add_argument('--seed',     type=int,   default=42,     help='RNG seed')
    args = ap.parse_args()

    random.seed(args.seed)

    # Load cameras; fall back to built-in simulation cameras if config is unusable
    cameras = load_cameras(args.cameras)
    print(f"[sim] loaded {len(cameras)} camera(s) from {args.cameras}")

    if not any(camera_sees_floor(c) for c in cameras):
        print("[sim] WARNING: configured cameras cannot see the floor "
              "(cam Z points up -- likely wrong pitch sign in placeholder config).")
        print("[sim] Using built-in simulation cameras (pitch=45 deg downward).")
        cameras = make_sim_cameras()
        for c in cameras:
            print(f"      sim cam {c['id']}: tx={c['tx']:.2f} ty={c['ty']:.2f} tz={c['tz']:.2f}")

    # Robot drives along +X at constant speed; cameras sweep over a mix of
    # static and moving objects with varied class IDs.
    robot_speed   = 0.4     # m/s along +X
    robot_heading = 0.0     # always facing +X
    drive_len     = robot_speed * args.duration  # total X travel

    # Camera lateral offsets (used to place objects in each camera's swept lane)
    cam_tys = [c['ty'] for c in cameras]
    if not cam_tys:
        cam_tys = [0.15, -0.15]

    objects = []
    oid = 0

    # 1. Static objects at 25 / 55 / 80 % along path, alternating lanes + class IDs
    for frac, cls in zip((0.25, 0.55, 0.80), (0, 1, 0)):
        ox = drive_len * frac
        for ty in cam_tys:
            objects.append(SimObject(oid, cls, x0=ox, y0=ty))
            oid += 1

    # 2. Moving object: crosses the robot's path diagonally (class 1)
    #    Starts behind the robot's starting X, moves at +0.15 m/s in X and +0.2 m/s in Y
    objects.append(SimObject(oid, 1, x0=drive_len * 0.35, y0=-0.6, vx=0.0, vy=0.25))
    oid += 1

    # 3. Moving object: drifts parallel to robot, slightly faster (class 2)
    objects.append(SimObject(oid, 2, x0=drive_len * 0.1, y0=cam_tys[0] * 0.5,
                             vx=robot_speed * 0.6, vy=0.0))
    oid += 1

    # 4. Two objects close together — separated enough to be individually gatable (class 0)
    cluster_x = drive_len * 0.68
    objects.append(SimObject(oid, 0, x0=cluster_x,        y0=0.30))
    oid += 1
    objects.append(SimObject(oid, 0, x0=cluster_x + 0.20, y0=-0.30))
    oid += 1

    print(f"[sim] {len(objects)} objects ({sum(1 for o in objects if o.vx==0 and o.vy==0)} static,"
          f" {sum(1 for o in objects if o.vx!=0 or o.vy!=0)} moving):")
    for o in objects:
        moving = f" v=({o.vx:.2f},{o.vy:.2f})" if (o.vx or o.vy) else " static"
        print(f"      obj {o.obj_id} cls={o.class_id}: field ({o.x0:.2f}, {o.y0:.2f}){moving}")

    # Timing
    pose_dt_ns  = int(1e9 / args.pose_hz)
    frame_dt_ns = int(1e9 / args.fps)
    total_ns    = int(args.duration * 1e9)
    base_ns     = int(time.time() * 1e9)

    # Build interleaved event list
    events = []  # (t_ns, 'pose'|'frame', payload)

    # Pose events
    t_ns = 0
    while t_ns <= total_ns:
        rx = robot_speed * (t_ns / 1e9)
        events.append((t_ns, 'pose', {
            't': 'pose', 'recv_ns': base_ns + t_ns,
            'x': rx, 'y': 0.0, 'heading': robot_heading, 'vyaw': 0.0,
            'ts_ns': base_ns + t_ns,
        }))
        t_ns += pose_dt_ns

    # Detection frame events + ground truth
    ground_truth = []
    t_ns = 0
    det_total = 0
    while t_ns <= total_ns:
        t_s     = t_ns / 1e9
        robot_x = robot_speed * t_s
        dets    = []
        gt      = []

        for obj in objects:
            ox, oy = obj.position(t_s)
            gt.append({'obj_id': obj.obj_id, 'class_id': obj.class_id,
                       'x': ox, 'y': oy, 'vx': obj.vx, 'vy': obj.vy})

            for cam in cameras:
                bbox = field_to_pixel(cam, robot_x, 0.0, robot_heading, ox, oy)
                if bbox is None:
                    continue
                l, top, w, h = bbox
                n = args.noise
                dets.append({
                    'cam':    cam['id'],
                    'cls':    obj.class_id,
                    'conf':   max(0.0, min(1.0, args.conf + random.gauss(0, 0.02))),
                    'l':      l    + random.gauss(0, n),
                    'top':    top  + random.gauss(0, n),
                    'w':      max(2.0, w + random.gauss(0, n * 0.3)),
                    'h':      max(2.0, h + random.gauss(0, n * 0.3)),
                    'ts_ns':  base_ns + t_ns,
                    'cap_ns': base_ns + t_ns,
                })
                det_total += 1
                break  # one camera per object

        events.append((t_ns, 'frame', {'t': 'frame', 'dets': dets}))
        ground_truth.append({'ts_ns': base_ns + t_ns, 'objects': gt})
        t_ns += frame_dt_ns

    events.sort(key=lambda e: e[0])

    # Write files
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    rec_path = args.out + '_recording.jsonl'
    gt_path  = args.out + '_ground_truth.jsonl'

    with open(rec_path, 'w') as f:
        for _, _, payload in events:
            f.write(json.dumps(payload) + '\n')

    with open(gt_path, 'w') as f:
        for frame in ground_truth:
            f.write(json.dumps(frame) + '\n')

    n_poses  = sum(1 for e in events if e[1] == 'pose')
    n_frames = sum(1 for e in events if e[1] == 'frame')
    print(f"[sim] {n_poses} poses + {n_frames} frames ({det_total} total detections) -> {rec_path}")
    print(f"[sim] ground truth -> {gt_path}")
    print()
    print("Run replay:")
    print(f"  ./build/heimdall_replay {rec_path} {args.cameras} replay_out.jsonl")
    print(f"Then compare replay_out.jsonl against {gt_path}")


if __name__ == '__main__':
    main()
