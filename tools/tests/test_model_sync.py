"""
test_model_sync.py -- guardrail against Python/C++ camera-model drift.

The sim's camera model (tools/sim/projection.py) mirrors the C++ pose_estimator.
If they drift, every metric the sim reports becomes a lie. This test closes the
loop with the REAL binary: a stationary robot views static objects straight
ahead, we run heimdall_replay (local or docker), and assert the recovered track
positions land within tolerance of ground truth.

Run:
  python tools/tests/test_model_sync.py                 # auto: local binary, else docker
  python tools/tests/test_model_sync.py --runner docker
  python tools/tests/test_model_sync.py --runner local --replay <path>

Exit code 0 = pass, 1 = drift/fail, 2 = no runner available (skipped).
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
from sim import projection, scene  # noqa: E402
import sim_eval  # reuse runner selection  # noqa: E402


# Controlled scenario: stationary robot, static objects in the camera model's
# accurate regime (0.6-0.8 m ahead, positive-y side), zero pixel noise. This
# isolates systematic camera-model error from tracker noise and range effects.
#
# The test runs with --bypass-tracker so the pipeline output is the RAW pose
# projection (no JPDAF/Kalman smoothing, no confirmation lag). Because the sim's
# field->pixel forward model (including Brown-Conrady distortion) is the exact
# inverse of the C++ pose_estimator's pixel->field back-projection, the recovered
# position is bitwise-identical to ground truth (~0 mm). The tolerance is
# therefore tight: any real drift (a rotation/sign flip, a distortion mismatch,
# a bottom-center convention change) breaks the round-trip and shows up as cm-to-
# metre error or zero detections, all far above TOL_MAX_M.
SCENARIO = {
    'name': 'model_sync',
    'duration': 4.0, 'fps': 30.0, 'pose_hz': 50.0,
    'noise': 0.0, 'conf': 0.95, 'seed': 1,
    'robot': {'type': 'linear', 'speed': 0.0, 'heading': 0.0},
    'objects': [
        {'class_id': 0, 'x': 0.6, 'y': 0.0},
        {'class_id': 0, 'x': 0.8, 'y': 0.0},
        {'class_id': 0, 'x': 0.8, 'y': 0.2},
    ],
    'apriltags': [],
}

# Projection-only (bypass tracker) round-trips to <1 cm; leave headroom for FP
# rounding across the JSON boundary. A broken model overshoots this by 10-100x.
TOL_MEAN_M = 0.01
TOL_MAX_M = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runner', choices=['auto', 'local', 'docker'], default='auto')
    ap.add_argument('--replay', default=None)
    ap.add_argument('--cameras', default='config/cameras')
    args = ap.parse_args()

    # Pick a runner exactly like sim_eval.
    local = sim_eval.find_local_replay(args.replay)
    runner = args.runner
    if runner == 'auto':
        runner = 'local' if local else ('docker' if sim_eval.docker_available() else None)
    if runner is None or (runner == 'local' and not local) or \
       (runner == 'docker' and not sim_eval.docker_available()):
        print("SKIP: no heimdall_replay runner available (build it or start Docker).")
        return 2

    cameras, fallback = projection.load_cameras_or_sim(args.cameras)
    if fallback:
        print("[test] note: using built-in sim cameras (config can't see floor)")

    # Must live under recordings/ so the docker runner can mount it.
    prefix = os.path.join('recordings', '_modelsync')
    stats = scene.generate(SCENARIO, cameras, os.path.join(REPO, prefix))
    out = os.path.join(REPO, prefix + '_out.jsonl')

    # Bypass the tracker: we are validating the projection model, not JPDAF.
    if runner == 'local':
        rc = sim_eval.run_local(local, stats['recording'], args.cameras, out, True)
    else:
        rc = sim_eval.run_docker(stats['recording'], args.cameras, out, True)
    if rc != 0:
        print(f"FAIL: heimdall_replay ({runner}) exited {rc}")
        return 1

    gt_rows = [json.loads(l) for l in open(stats['ground_truth']) if l.strip()]
    out_rows = [json.loads(l) for l in open(out) if l.strip()]
    if not out_rows:
        print("FAIL: replay produced no output rows")
        return 1

    gt_last = {int(o['obj_id']): (float(o['x']), float(o['y'])) for o in gt_rows[-1]['objects']}
    last_track = {}
    for row in out_rows:
        for ev in row.get('events', []):
            if 'track_id' in ev and 'x' in ev:
                last_track[int(ev['track_id'])] = (float(ev['x']), float(ev['y']))
    if not last_track:
        print("FAIL: no tracks confirmed -- pipeline saw nothing (model broken?)")
        return 1

    errs = []
    for oid, (gx, gy) in gt_last.items():
        best = min((math.hypot(tx - gx, ty - gy) for (tx, ty) in last_track.values()),
                   default=float('inf'))
        errs.append(best)
        print(f"  GT obj {oid} at ({gx:+.2f},{gy:+.2f})  nearest-track err = {best:.3f} m")

    mean_err, max_err = sum(errs) / len(errs), max(errs)
    print(f"[test] mean err {mean_err:.3f} m (tol {TOL_MEAN_M}), "
          f"max err {max_err:.3f} m (tol {TOL_MAX_M})")

    if mean_err <= TOL_MEAN_M and max_err <= TOL_MAX_M:
        print("PASS: Python projector agrees with C++ pose_estimator within tolerance.")
        return 0
    print("FAIL: model drift -- Python projection disagrees with C++ pose_estimator.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
