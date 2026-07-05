"""
scene.py -- scenario loading, kinematics, and recording/ground-truth emit.

A scenario is a .jsonc file describing a 3D field setup: a robot trajectory, a
set of "fuel" game pieces (static or moving), and AprilTag placements. This
module turns a scenario + camera configs into the two JSONL artifacts the rest
of the pipeline already understands:

  <out>_recording.jsonl     poses + noisy detections   -> heimdall_replay
  <out>_ground_truth.jsonl  perfect field positions    -> replay_metrics.py

The camera model comes entirely from projection.py -- this module never does
its own geometry. AprilTags are recorded into the scenario but NOT projected in
v1 (tag image detection is a video-level / Phase 3 concern); they are carried so
future renderers and metrics can consume the same scenario file.
"""
import json
import os
import random

from . import projection


# ── Scenario loading ───────────────────────────────────────────────────────────

# Default simulation parameters, overridable per-scenario and via CLI.
DEFAULTS = {
    'duration': 10.0,   # seconds
    'fps':      30.0,   # detection frame rate
    'pose_hz':  50.0,   # pose publish rate
    'noise':    5.0,    # pixel noise std-dev
    'conf':     0.85,   # base detection confidence
    'seed':     42,
    # Fixed timestamp base (ns) so runs are BITWISE reproducible. Wall-clock
    # time here would make absolute ts differ between runs, perturbing Kalman dt
    # at float64 precision and breaking determinism. Override only if you need a
    # realistic epoch; relative timing (the only thing replay uses) is unaffected.
    'base_ns':  1_700_000_000_000_000_000,
}


def load_scenario(path):
    """Load a .jsonc scenario, stripping // comments (same as camera configs)."""
    with open(path, encoding='utf-8') as f:
        scn = json.loads(projection.strip_comments(f.read()))
    return scn


def default_scenario(cameras, duration=DEFAULTS['duration']):
    """Reproduce the original hardcoded sim_recording.py scene as a scenario dict.

    Used when the CLI is invoked without --scenario, so existing behavior is
    preserved. Robot drives +X at 0.4 m/s; objects are laid out relative to the
    cameras' lateral offsets, matching the pre-refactor generator.
    """
    robot_speed = 0.4
    drive_len   = robot_speed * duration

    cam_tys = [c['ty'] for c in cameras] or [0.15, -0.15]

    objects = []
    # 1. Static objects at 25 / 55 / 80 % along path, alternating lanes + class IDs
    for frac, cls in zip((0.25, 0.55, 0.80), (0, 1, 0)):
        ox = drive_len * frac
        for ty in cam_tys:
            objects.append({'class_id': cls, 'x': ox, 'y': ty, 'vx': 0.0, 'vy': 0.0})
    # 2. Moving object: crosses the robot's path diagonally (class 1)
    objects.append({'class_id': 1, 'x': drive_len * 0.35, 'y': -0.6, 'vx': 0.0, 'vy': 0.25})
    # 3. Moving object: drifts parallel to robot, slightly faster (class 2)
    objects.append({'class_id': 2, 'x': drive_len * 0.1, 'y': cam_tys[0] * 0.5,
                    'vx': robot_speed * 0.6, 'vy': 0.0})
    # 4. Two objects close together -- separated enough to be individually gatable (class 0)
    cluster_x = drive_len * 0.68
    objects.append({'class_id': 0, 'x': cluster_x,        'y':  0.30, 'vx': 0.0, 'vy': 0.0})
    objects.append({'class_id': 0, 'x': cluster_x + 0.20, 'y': -0.30, 'vx': 0.0, 'vy': 0.0})

    return {
        'name':     'default (hardcoded legacy scene)',
        'duration': duration,
        'robot':    {'type': 'linear', 'speed': robot_speed, 'heading': 0.0},
        'objects':  objects,
        'apriltags': [],
    }


# ── Robot trajectory ───────────────────────────────────────────────────────────

def robot_pose_at(robot, t):
    """Evaluate robot pose (x, y, heading, vyaw) at time t seconds.

    Supported robot types:
      "linear":    {speed, heading}         -- drive along heading at speed.
      "waypoints": {waypoints:[{t,x,y,heading}, ...]}  -- piecewise-linear interp.
    """
    kind = robot.get('type', 'linear')

    if kind == 'linear':
        speed   = float(robot.get('speed', 0.4))
        heading = float(robot.get('heading', 0.0))
        import math
        return (speed * t * math.cos(heading), speed * t * math.sin(heading), heading, 0.0)

    if kind == 'waypoints':
        wps = robot['waypoints']
        if not wps:
            return (0.0, 0.0, 0.0, 0.0)
        if t <= wps[0]['t']:
            w = wps[0]
            return (float(w['x']), float(w['y']), float(w.get('heading', 0.0)), 0.0)
        for a, b in zip(wps, wps[1:]):
            ta, tb = float(a['t']), float(b['t'])
            if ta <= t <= tb:
                span = tb - ta
                f = 0.0 if span <= 0 else (t - ta) / span
                x = float(a['x']) + f * (float(b['x']) - float(a['x']))
                y = float(a['y']) + f * (float(b['y']) - float(a['y']))
                ha, hb = float(a.get('heading', 0.0)), float(b.get('heading', 0.0))
                heading = ha + f * (hb - ha)
                vyaw = 0.0 if span <= 0 else (hb - ha) / span
                return (x, y, heading, vyaw)
        w = wps[-1]
        return (float(w['x']), float(w['y']), float(w.get('heading', 0.0)), 0.0)

    raise ValueError(f"unknown robot trajectory type: {kind!r}")


# ── Generation ─────────────────────────────────────────────────────────────────

def generate(scenario, cameras, out_prefix, overrides=None):
    """Run a scenario through the camera model and write the two JSONL files.

    overrides: optional dict of {duration,fps,pose_hz,noise,conf,seed} that wins
    over scenario values (used to plumb CLI flags through).
    Returns a stats dict.
    """
    overrides = overrides or {}

    def param(key):
        if key in overrides and overrides[key] is not None:
            return overrides[key]
        if key in scenario and scenario[key] is not None:
            return scenario[key]
        return DEFAULTS[key]

    duration = float(param('duration'))
    fps      = float(param('fps'))
    pose_hz  = float(param('pose_hz'))
    noise    = float(param('noise'))
    conf     = float(param('conf'))
    seed     = int(param('seed'))

    random.seed(seed)

    robot   = scenario.get('robot', {'type': 'linear', 'speed': 0.4, 'heading': 0.0})
    objects = scenario.get('objects', [])

    # Timing
    pose_dt_ns  = int(1e9 / pose_hz)
    frame_dt_ns = int(1e9 / fps)
    total_ns    = int(duration * 1e9)
    base_ns     = int(param('base_ns'))

    events = []  # (t_ns, 'pose'|'frame', payload)

    # Pose events
    t_ns = 0
    while t_ns <= total_ns:
        rx, ry, heading, vyaw = robot_pose_at(robot, t_ns / 1e9)
        events.append((t_ns, 'pose', {
            't': 'pose', 'recv_ns': base_ns + t_ns,
            'x': rx, 'y': ry, 'heading': heading, 'vyaw': vyaw,
            'ts_ns': base_ns + t_ns,
        }))
        t_ns += pose_dt_ns

    # Detection frame events + ground truth
    ground_truth = []
    det_total = 0
    t_ns = 0
    while t_ns <= total_ns:
        t_s = t_ns / 1e9
        rx, ry, heading, _ = robot_pose_at(robot, t_s)
        dets = []
        gt   = []

        for oid, obj in enumerate(objects):
            cls    = int(obj.get('class_id', 0))
            radius = float(obj.get('radius', 0.12))
            ox = float(obj['x']) + float(obj.get('vx', 0.0)) * t_s
            oy = float(obj['y']) + float(obj.get('vy', 0.0)) * t_s
            gt.append({'obj_id': oid, 'class_id': cls, 'x': ox, 'y': oy,
                       'vx': float(obj.get('vx', 0.0)), 'vy': float(obj.get('vy', 0.0))})

            for cam in cameras:
                bbox = projection.field_to_pixel(cam, rx, ry, heading, ox, oy, radius)
                if bbox is None:
                    continue
                l, top, w, h = bbox
                dets.append({
                    'cam':    cam['id'],
                    'cls':    cls,
                    'conf':   max(0.0, min(1.0, conf + random.gauss(0, 0.02))),
                    'l':      l    + random.gauss(0, noise),
                    'top':    top  + random.gauss(0, noise),
                    'w':      max(2.0, w + random.gauss(0, noise * 0.3)),
                    'h':      max(2.0, h + random.gauss(0, noise * 0.3)),
                    'ts_ns':  base_ns + t_ns,
                    'cap_ns': base_ns + t_ns,
                })
                det_total += 1
                break  # one camera per object

        events.append((t_ns, 'frame', {'t': 'frame', 'dets': dets}))
        ground_truth.append({'ts_ns': base_ns + t_ns, 'objects': gt})
        t_ns += frame_dt_ns

    events.sort(key=lambda e: e[0])

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    rec_path = out_prefix + '_recording.jsonl'
    gt_path  = out_prefix + '_ground_truth.jsonl'

    with open(rec_path, 'w') as f:
        for _, _, payload in events:
            f.write(json.dumps(payload) + '\n')
    with open(gt_path, 'w') as f:
        for frame in ground_truth:
            f.write(json.dumps(frame) + '\n')

    return {
        'recording':  rec_path,
        'ground_truth': gt_path,
        'n_poses':   sum(1 for e in events if e[1] == 'pose'),
        'n_frames':  sum(1 for e in events if e[1] == 'frame'),
        'n_dets':    det_total,
        'n_objects': len(objects),
        'n_apriltags': len(scenario.get('apriltags', [])),
    }
