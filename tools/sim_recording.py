"""
sim_recording.py -- Synthetic record/replay dataset generator (thin CLI)

Simulates a 3D field scene (robot + "fuel" pieces) in field space, projects it to
noisy pixel detections, and writes two files:
  <out>_recording.jsonl     -- noisy detections + poses  (feed to heimdall_replay)
  <out>_ground_truth.jsonl  -- perfect field positions    (diff target)

The camera model and scene kinematics now live in the tools/sim package:
  tools/sim/projection.py   -- single source of truth for the camera geometry
  tools/sim/scene.py        -- scenario loading, kinematics, JSONL emit

Usage:
  # Explicit scenario file:
  python tools/sim_recording.py --scenario tools/scenarios/basic.jsonc \
      --cameras config/cameras --out recordings/basic

  # No --scenario: reproduces the original hardcoded scene (backward compatible):
  python tools/sim_recording.py --cameras config/cameras --out recordings/sim

Camera config dir is read exactly like the live system (.jsonc with // comments).
If the configured cameras cannot see the floor (e.g. placeholder pitch sign is
wrong), the sim falls back to built-in simulation cameras and prints a warning.
"""
import argparse
import os
import sys

# Allow running as a plain script (python tools/sim_recording.py) as well as a
# module; make the tools/ dir importable so `from sim import ...` resolves.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import projection, scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic recording for heimdall_replay")
    ap.add_argument('--scenario', default=None,
                    help='Scenario .jsonc file (omit to use the legacy hardcoded scene)')
    ap.add_argument('--cameras',  default='config/cameras', help='Camera config dir (.jsonc files)')
    ap.add_argument('--out',      default='recordings/sim',  help='Output path prefix (no extension)')
    # Overrides (win over scenario/defaults when provided).
    ap.add_argument('--duration', type=float, default=None, help='Simulation duration (seconds)')
    ap.add_argument('--fps',      type=float, default=None, help='Detection frame rate')
    ap.add_argument('--pose-hz',  type=float, default=None, help='Pose publish rate')
    ap.add_argument('--noise',    type=float, default=None, help='Pixel noise std-dev (px)')
    ap.add_argument('--conf',     type=float, default=None, help='Detection confidence')
    ap.add_argument('--seed',     type=int,   default=None, help='RNG seed')
    args = ap.parse_args()

    # Load cameras; fall back to built-in simulation cameras if config is unusable.
    cameras, used_fallback = projection.load_cameras_or_sim(args.cameras)
    print(f"[sim] loaded {len(cameras)} camera(s) from {args.cameras}")
    if used_fallback:
        print("[sim] WARNING: configured cameras cannot see the floor "
              "(cam Z points up -- likely wrong pitch sign in placeholder config).")
        print("[sim] Using built-in simulation cameras (pitch=45 deg downward).")
        for c in cameras:
            print(f"      sim cam {c['id']}: tx={c['tx']:.2f} ty={c['ty']:.2f} tz={c['tz']:.2f}")

    # Load or synthesize the scenario.
    if args.scenario:
        scn = scene.load_scenario(args.scenario)
        print(f"[sim] scenario '{scn.get('name', args.scenario)}' from {args.scenario}")
    else:
        duration = args.duration if args.duration is not None else scene.DEFAULTS['duration']
        scn = scene.default_scenario(cameras, duration)
        print("[sim] no --scenario given; using legacy hardcoded scene")

    overrides = {
        'duration': args.duration, 'fps': args.fps, 'pose_hz': args.pose_hz,
        'noise': args.noise, 'conf': args.conf, 'seed': args.seed,
    }

    objs = scn.get('objects', [])
    n_moving = sum(1 for o in objs if o.get('vx') or o.get('vy'))
    print(f"[sim] {len(objs)} objects ({len(objs) - n_moving} static, {n_moving} moving),"
          f" {len(scn.get('apriltags', []))} apriltag(s)")

    stats = scene.generate(scn, cameras, args.out, overrides)

    print(f"[sim] {stats['n_poses']} poses + {stats['n_frames']} frames "
          f"({stats['n_dets']} total detections) -> {stats['recording']}")
    print(f"[sim] ground truth -> {stats['ground_truth']}")
    print()
    print("Run replay:")
    print(f"  ./build/heimdall_replay {stats['recording']} {args.cameras} replay_out.jsonl")
    print(f"Then score it:")
    print(f"  python tools/replay_metrics.py replay_out.jsonl {stats['ground_truth']}")


if __name__ == '__main__':
    main()
