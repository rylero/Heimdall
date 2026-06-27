"""
compare_replay.py -- Animate ground truth vs tracker output from a sim recording.

Loads sim_recording.jsonl and sim_ground_truth.jsonl, runs a Python replay
(project pixel->field + simple Kalman NN tracker), then plays back an
animated top-down comparison.

Legend:
  Green filled circles   -- ground truth object positions
  Colored filled circles -- tracker estimates (one colour per track ID)
  Grey dots              -- raw projected detections (noisy)
  Blue square            -- robot position
  Dashed circle          -- tracker gate radius

Controls:
  Space      -- pause / resume
  Left/Right -- step one frame
  Scroll     -- zoom
  Q / Esc    -- quit

Usage:
  python tools/compare_replay.py [recording.jsonl] [ground_truth.jsonl]
"""

import json, math, sys, time, re, os, argparse

try:
    import pygame
except ImportError:
    sys.exit("pip install pygame")


# ── Linear algebra (no numpy) ─────────────────────────────────────────────────

def mat3_mul(A, B):
    C = [0.0]*9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i*3+j] += A[i*3+k]*B[k*3+j]
    return C

def mat3T(M):
    return [M[0],M[3],M[6], M[1],M[4],M[7], M[2],M[5],M[8]]

def mv(M, v):
    return (M[0]*v[0]+M[1]*v[1]+M[2]*v[2],
            M[3]*v[0]+M[4]*v[1]+M[5]*v[2],
            M[6]*v[0]+M[7]*v[1]+M[8]*v[2])

def rotation_from_euler(yaw, pitch, roll):
    R_base = [0,0,1, -1,0,0, 0,-1,0]
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    Rx = [1,0,0, 0,cp,-sp, 0,sp,cp]
    cr, sr = math.cos(roll), math.sin(roll)
    Rz_cam = [cr,-sr,0, sr,cr,0, 0,0,1]
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz_rob = [cy,-sy,0, sy,cy,0, 0,0,1]
    return mat3_mul(Rz_rob, mat3_mul(mat3_mul(R_base, Rx), Rz_cam))


# ── Camera config ─────────────────────────────────────────────────────────────

def camera_sees_floor(cam):
    return cam['R_rob_to_cam'][8] < -0.1

def make_sim_cameras():
    cams = []
    for i, (tx, ty) in enumerate([(0.20, 0.15), (0.20, -0.15)]):
        R = rotation_from_euler(0.0, math.radians(70.0), 0.0)
        cams.append({'id': i, 'width': 640, 'height': 480,
                     'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0,
                     'tx': tx, 'ty': ty, 'tz': 0.30,
                     'R_rob_to_cam': mat3T(R)})
    return cams

def load_cameras(cam_dir):
    def strip(t): return re.sub(r'//[^\n]*', '', t)
    paths = sorted(os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.jsonc'))
    cams = []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            cfg = json.loads(strip(f.read()))
        intr, extr = cfg['intrinsics'], cfg['extrinsics']
        R = rotation_from_euler(float(extr['yaw']), float(extr['pitch']), float(extr['roll']))
        cams.append({'id': cfg['id'], 'width': cfg.get('width',640), 'height': cfg.get('height',480),
                     'fx': float(intr['fx']), 'fy': float(intr['fy']),
                     'cx': float(intr['cx']), 'cy': float(intr['cy']),
                     'tx': float(extr['tx']), 'ty': float(extr['ty']), 'tz': float(extr['tz']),
                     'R_rob_to_cam': mat3T(R)})
    return cams


# ── Fudge factor (mirrors pose_estimator.cpp) ────────────────────────────────
_kA, _kB, _kC = -0.0658, 0.9637, 0.12

def apply_fudge(fx, fy):
    """Same correction as pose_estimator.cpp — applied after geometric projection."""
    raw_d = math.hypot(fx, fy)
    if raw_d < 0.01:
        return fx, fy
    corr_d = max(0.0, (_kA * raw_d + _kB) * raw_d + _kC)
    s = corr_d / raw_d
    return fx * s, fy * s


# ── Projection: pixel -> field (mirrors pose_estimator.cpp) ──────────────────

def pixel_to_field(cam, px, py, robot_x, robot_y, robot_heading):
    """Unproject pixel to field ground plane, including fudge factor. Returns (fx, fy) or None."""
    u = (px - cam['cx']) / cam['fx']
    v = (py - cam['cy']) / cam['fy']

    R_cam_rob = mat3T(cam['R_rob_to_cam'])
    d_rob = mv(R_cam_rob, (u, v, 1.0))

    cos_h, sin_h = math.cos(robot_heading), math.sin(robot_heading)
    dfx = cos_h * d_rob[0] - sin_h * d_rob[1]
    dfy = sin_h * d_rob[0] + cos_h * d_rob[1]
    dfz = d_rob[2]

    if dfz >= 0:
        return None

    ofx = robot_x + cos_h * cam['tx'] - sin_h * cam['ty']
    ofy = robot_y + sin_h * cam['tx'] + cos_h * cam['ty']

    t = -cam['tz'] / dfz
    fx = ofx + t * dfx
    fy = ofy + t * dfy
    return apply_fudge(fx, fy)


# ── Simple Kalman tracker ─────────────────────────────────────────────────────
# State: [x, y, vx, vy]. No JPDAF — uses greedy nearest-neighbour assignment.

TRACK_COLORS = [
    (0, 200, 255), (255, 180, 0), (0, 255, 120), (255, 80, 80),
    (200, 0, 255), (255, 220, 0), (0, 255, 220), (255, 140, 200),
]

class Track:
    Q = 2.0   # process noise
    R = 0.16  # measurement noise
    GATE = 1.2  # metres

    def __init__(self, tid, x, y):
        self.id   = tid
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.pxx = self.pyy = 0.5
        self.hits = 1
        self.miss = 0

    def predict(self, dt):
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.pxx += self.Q * dt * dt
        self.pyy += self.Q * dt * dt

    def update(self, mx, my):
        kx = self.pxx / (self.pxx + self.R)
        ky = self.pyy / (self.pyy + self.R)
        innov_x = mx - self.x;  innov_y = my - self.y
        # Velocity smoothed from position innovation
        alpha = 0.3
        self.vx = (1 - alpha) * self.vx + alpha * innov_x
        self.vy = (1 - alpha) * self.vy + alpha * innov_y
        self.x  += kx * innov_x
        self.y  += ky * innov_y
        self.pxx *= (1 - kx)
        self.pyy *= (1 - ky)
        self.hits += 1
        self.miss  = 0


class SimpleTracker:
    CONFIRM = 2   # hits before confirmed
    LOSE    = 5   # misses before removed

    def __init__(self):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.last_ts = None

    def update(self, detections, timestamp_s):
        dt = (timestamp_s - self.last_ts) if self.last_ts else 0.033
        self.last_ts = timestamp_s

        for t in self.tracks:
            t.predict(dt)

        assigned = set()
        for t in self.tracks:
            best_d, best_i = float('inf'), None
            for i, (dx, dy) in enumerate(detections):
                if i in assigned: continue
                d = math.hypot(t.x - dx, t.y - dy)
                if d < Track.GATE and d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                t.update(*detections[best_i])
                assigned.add(best_i)
            else:
                t.miss += 1

        for i, (dx, dy) in enumerate(detections):
            if i not in assigned:
                t = Track(self.next_id, dx, dy)
                self.next_id += 1
                self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.miss < self.LOSE]

        return [t for t in self.tracks if t.hits >= self.CONFIRM]


# ── Replay engine ─────────────────────────────────────────────────────────────

def run_replay(recording_path, cameras):
    """
    Load recording, project detections to field space, run simple tracker.
    Returns list of replay frames: [{ts_ns, robot_x, robot_y, raw_dets, tracks}]
    """
    with open(recording_path) as f:
        lines = f.readlines()

    poses  = {}   # recv_ns -> (x, y, heading)
    frames = []

    for line in lines:
        j = json.loads(line)
        t = j['t']
        if t == 'pose':
            poses[j['recv_ns']] = (j['x'], j['y'], j['heading'])
        elif t == 'frame':
            frames.append(j)

    # Build sorted pose list for closest-lookup
    pose_list = sorted(poses.items())  # [(recv_ns, (x,y,h)), ...]

    def closest_pose(target_ns):
        if not pose_list: return 0.0, 0.0, 0.0
        best = pose_list[0][1]
        best_d = abs(pose_list[0][0] - target_ns)
        for recv_ns, p in pose_list:
            d = abs(recv_ns - target_ns)
            if d < best_d:
                best_d, best = d, p
            elif recv_ns > target_ns + best_d:
                break
        return best

    tracker = SimpleTracker()
    replay_frames = []

    for frame in frames:
        dets  = frame['dets']
        # Use cap_ns from first detection; fall back to last known ts so timing stays monotonic
        if dets:
            ts_ns = dets[0]['cap_ns']
        else:
            ts_ns = replay_frames[-1]['ts_ns'] if replay_frames else 0

        if not dets:
            pose = closest_pose(ts_ns) if ts_ns else (0.0, 0.0, 0.0)
            replay_frames.append({'ts_ns': ts_ns, 'robot': pose, 'raw': [], 'tracks': []})
            continue
        pose  = closest_pose(ts_ns)
        rx, ry, rh = pose

        raw = []
        for d in dets:
            cam = next((c for c in cameras if c['id'] == d['cam']), None)
            if cam is None: continue
            # pose_estimator uses bottom-centre: px = l+w/2, py = top+h
            px = d['l'] + d['w'] / 2
            py = d['top'] + d['h']
            pt = pixel_to_field(cam, px, py, rx, ry, rh)
            if pt:
                raw.append(pt)

        ts_s   = ts_ns / 1e9
        tracks = tracker.update(raw, ts_s)
        replay_frames.append({
            'ts_ns':  ts_ns,
            'robot':  (rx, ry, rh),
            'raw':    raw,
            'tracks': [{'id': t.id, 'x': t.x, 'y': t.y, 'vx': t.vx, 'vy': t.vy} for t in tracks],
        })

    return replay_frames


# ── Visualizer ────────────────────────────────────────────────────────────────

WIN_W, WIN_H = 1280, 720
SCALE0 = 120   # pixels per metre at default zoom

class Viz:
    def __init__(self, replay_frames, gt_frames):
        self.replay = replay_frames
        self.gt     = gt_frames
        self.n      = max(len(replay_frames), len(gt_frames))
        self.idx    = 0
        self.paused = False
        self.scale  = SCALE0
        self.pan_x  = WIN_W // 2
        self.pan_y  = WIN_H // 2
        self.t_last = time.monotonic()
        self.gt_map = {f['ts_ns']: f for f in gt_frames}

    def field_to_screen(self, fx, fy):
        sx = int(self.pan_x + fx * self.scale)
        sy = int(self.pan_y - fy * self.scale)
        return sx, sy

    def advance(self):
        if self.paused or self.idx >= self.n - 1:
            return
        if len(self.replay) < 2:
            return
        now = time.monotonic()
        elapsed = now - self.t_last

        ri  = min(self.idx,     len(self.replay) - 1)
        ri1 = min(self.idx + 1, len(self.replay) - 1)
        dt_ns = self.replay[ri1]['ts_ns'] - self.replay[ri]['ts_ns']
        # Fall back to 33ms if timestamps are identical or zero (e.g. empty frames)
        frame_dt_s = dt_ns / 1e9 if dt_ns > 0 else 1.0 / 30.0

        if elapsed >= frame_dt_s:
            self.idx   = min(self.idx + 1, self.n - 1)
            # Advance anchor by one frame period — keeps phase stable, avoids drift
            self.t_last += frame_dt_s

    def draw(self, surf, font, big_font):
        surf.fill((40, 60, 40))

        # Grid lines (1m spacing)
        for x in range(-5, 20):
            sx0, sy0 = self.field_to_screen(x, -5)
            sx1, sy1 = self.field_to_screen(x, 15)
            pygame.draw.line(surf, (55, 75, 55), (sx0, sy0), (sx1, sy1), 1)
        for y in range(-5, 15):
            sx0, sy0 = self.field_to_screen(-5, y)
            sx1, sy1 = self.field_to_screen(20, y)
            pygame.draw.line(surf, (55, 75, 55), (sx0, sy0), (sx1, sy1), 1)

        # Origin axes
        ox, oy = self.field_to_screen(0, 0)
        pygame.draw.line(surf, (100, 100, 200), self.field_to_screen(-0.3,0), self.field_to_screen(0.8,0), 2)
        pygame.draw.line(surf, (100, 200, 100), self.field_to_screen(0,-0.3), self.field_to_screen(0,0.8), 2)

        ri = min(self.idx, len(self.replay)-1)
        rf = self.replay[ri]

        # ── Robot ─────────────────────────────────────────────────────────────
        rx, ry, rh = rf['robot']
        rsx, rsy = self.field_to_screen(rx, ry)
        pygame.draw.rect(surf, (50, 160, 255), pygame.Rect(rsx-10, rsy-10, 20, 20))
        pygame.draw.rect(surf, (255,255,255), pygame.Rect(rsx-10, rsy-10, 20, 20), 2)
        # heading arrow
        ax = int(rsx + 14 * math.cos(rh))
        ay = int(rsy - 14 * math.sin(rh))
        pygame.draw.line(surf, (255,255,255), (rsx, rsy), (ax, ay), 2)

        # ── Ground truth ──────────────────────────────────────────────────────
        # Find closest GT frame by timestamp
        if self.gt:
            closest_gt = min(self.gt, key=lambda f: abs(f['ts_ns'] - rf['ts_ns']))
            for obj in closest_gt.get('objects', []):
                gx, gy = self.field_to_screen(obj['x'], obj['y'])
                pygame.draw.circle(surf, (0, 230, 80), (gx, gy), 12)
                pygame.draw.circle(surf, (200, 255, 200), (gx, gy), 12, 2)
                lbl = font.render(f"GT{obj['obj_id']}", True, (200, 255, 200))
                surf.blit(lbl, (gx+14, gy-8))

        # ── Raw projected detections (noisy) ─────────────────────────────────
        for (dx, dy) in rf['raw']:
            sx, sy = self.field_to_screen(dx, dy)
            pygame.draw.circle(surf, (180, 180, 180), (sx, sy), 4)
            pygame.draw.circle(surf, (255, 255, 255), (sx, sy), 4, 1)

        # ── Tracker estimates ─────────────────────────────────────────────────
        for tr in rf['tracks']:
            color = TRACK_COLORS[tr['id'] % len(TRACK_COLORS)]
            tx2, ty2 = self.field_to_screen(tr['x'], tr['y'])

            # Gate circle (dashed approximation)
            gate_r = int(Track.GATE * self.scale)
            pygame.draw.circle(surf, tuple(c//3 for c in color), (tx2, ty2), gate_r, 1)

            # Velocity arrow
            vex = int(tx2 + tr['vx'] * 0.5 * self.scale)
            vey = int(ty2 - tr['vy'] * 0.5 * self.scale)
            pygame.draw.line(surf, color, (tx2, ty2), (vex, vey), 2)

            # Track circle
            pygame.draw.circle(surf, color, (tx2, ty2), 10)
            pygame.draw.circle(surf, (255,255,255), (tx2, ty2), 10, 2)

            lbl = font.render(f"T{tr['id']}", True, (255,255,255))
            surf.blit(lbl, (tx2+12, ty2-8))

        # ── Legend ────────────────────────────────────────────────────────────
        ly = 10
        for color, label in [
            ((0,230,80),      "Ground truth (GT)"),
            ((200,200,200),   "Raw detection (noisy projected)"),
            ((0,200,255),     "Tracker estimate"),
            ((50,160,255),    "Robot"),
        ]:
            pygame.draw.circle(surf, color, (20, ly+6), 6)
            surf.blit(font.render(label, True, (220,220,220)), (32, ly))
            ly += 18

        # ── HUD ───────────────────────────────────────────────────────────────
        progress = self.idx / max(1, self.n-1)
        ts_s = rf['ts_ns'] / 1e9
        n_tracks = len(rf['tracks'])
        n_raw    = len(rf['raw'])
        status   = "PAUSED" if self.paused else "PLAYING"
        hud = big_font.render(
            f"Frame {self.idx+1}/{self.n}  t={ts_s:.2f}s  "
            f"raw={n_raw}  tracks={n_tracks}  zoom={self.scale}  [{status}]",
            True, (220, 220, 100))
        surf.blit(hud, (10, WIN_H - 30))

        # Progress bar
        bar_w = int(progress * (WIN_W - 20))
        pygame.draw.rect(surf, (80,80,80),  pygame.Rect(10, WIN_H-10, WIN_W-20, 6))
        pygame.draw.rect(surf, (200,200,80), pygame.Rect(10, WIN_H-10, bar_w, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recording',    nargs='?', default='recordings/sim_recording.jsonl')
    ap.add_argument('ground_truth', nargs='?', default='recordings/sim_ground_truth.jsonl')
    ap.add_argument('--cameras',    default='config/cameras')
    args = ap.parse_args()

    # Load cameras — fall back to sim cameras if floor not visible
    cameras = load_cameras(args.cameras)
    if not any(camera_sees_floor(c) for c in cameras):
        print("[compare] Config cameras face up — using built-in sim cameras")
        cameras = make_sim_cameras()

    print(f"[compare] Running Python replay on {args.recording}...")
    replay_frames = run_replay(args.recording, cameras)
    print(f"[compare] {len(replay_frames)} replay frames")

    with open(args.ground_truth) as f:
        gt_frames = [json.loads(l) for l in f if l.strip()]
    print(f"[compare] {len(gt_frames)} ground truth frames")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Heimdall Replay Comparison")
    clock  = pygame.font.SysFont("monospace", 11)
    big    = pygame.font.SysFont("monospace", 14, bold=True)
    ticker = pygame.time.Clock()

    viz = Viz(replay_frames, gt_frames)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); return
                if ev.key == pygame.K_SPACE:
                    viz.paused = not viz.paused
                    viz.t_last = time.monotonic()
                if ev.key == pygame.K_RIGHT:
                    viz.idx = min(viz.idx + 1, viz.n - 1)
                if ev.key == pygame.K_LEFT:
                    viz.idx = max(viz.idx - 1, 0)
            if ev.type == pygame.MOUSEWHEEL:
                viz.scale = max(20, min(600, int(viz.scale * (1.15 if ev.y > 0 else 0.87))))

        viz.advance()
        viz.draw(screen, clock, big)
        pygame.display.flip()
        ticker.tick(60)


if __name__ == '__main__':
    main()
