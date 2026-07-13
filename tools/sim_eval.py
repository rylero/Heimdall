"""
sim_eval.py -- one command: scenario -> recording -> heimdall_replay -> metrics.

Validate a vision-pipeline change against a scenario in a single step:

  python tools/sim_eval.py --scenario tools/scenarios/basic.jsonc

It generates the recording + ground truth (via tools/sim), runs the real
heimdall_replay engine, then scores the output with replay_metrics.py.

Runner (--runner):
  auto   (default) use a local binary if found, else the docker image
  local  run ./build/heimdall_replay[.exe]  (needs a native build)
  docker run via docker/docker-compose.yml 'replay' service

On Windows, libzmq does not cross-compile under MinGW, so the docker runner is
the supported path (Ubuntu build). It mounts recordings/ and config/, so --out
must live under recordings/ and --cameras under config/.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from sim import projection, scene  # noqa: E402

COMPOSE_FILE = os.path.join('docker', 'docker-compose.yml')


def find_local_replay(explicit):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for cand in ('build/heimdall_replay.exe', 'build/heimdall_replay',
                 'build/Release/heimdall_replay.exe', 'build/Debug/heimdall_replay.exe'):
        p = os.path.join(REPO, cand)
        if os.path.exists(p):
            return p
    return None


def docker_available():
    try:
        return subprocess.run(['docker', 'version'], cwd=REPO,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        return False


def to_container_path(host_path, host_root_rel, container_root):
    """Map a host path under <repo>/<host_root_rel> to <container_root>/... .

    Used to translate --out (under recordings/) and --cameras (under config/) into
    the paths the mounted docker image sees.
    """
    abs_host = os.path.abspath(host_path)
    abs_root = os.path.abspath(os.path.join(REPO, host_root_rel))
    rel = os.path.relpath(abs_host, abs_root)
    if rel.startswith('..'):
        raise ValueError(
            f"docker runner needs {host_path!r} to live under {host_root_rel}/ "
            f"(it is mounted into the container); got outside it.")
    return container_root.rstrip('/') + '/' + rel.replace(os.sep, '/')


def run_local(binary, rec, cameras, out_jsonl, bypass):
    cmd = [binary, rec, cameras, out_jsonl] + (['--bypass-tracker'] if bypass else [])
    print(f"[eval] local: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO).returncode


def run_docker(rec, cameras, out_jsonl, bypass):
    c_rec = to_container_path(rec, 'recordings', '/recordings')
    c_out = to_container_path(out_jsonl, 'recordings', '/recordings')
    c_cam = to_container_path(cameras, 'config', '/app/config')
    cmd = ['docker', 'compose', '-f', COMPOSE_FILE, 'run', '--rm', 'replay',
           c_rec, c_cam, c_out] + (['--bypass-tracker'] if bypass else [])
    print(f"[eval] docker: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO).returncode


def main():
    ap = argparse.ArgumentParser(description="scenario -> replay -> metrics, in one command")
    ap.add_argument('--scenario', default='tools/scenarios/basic.jsonc')
    ap.add_argument('--cameras',  default='config/cameras')
    ap.add_argument('--out',      default='recordings/eval')
    ap.add_argument('--runner',   choices=['auto', 'local', 'docker'], default='auto')
    ap.add_argument('--replay',   default=None, help='path to local heimdall_replay binary')
    ap.add_argument('--gate',     type=float, default=0.75)
    ap.add_argument('--bypass-tracker', action='store_true')
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--noise',    type=float, default=None)
    ap.add_argument('--seed',     type=int,   default=None)
    ap.add_argument('--clutter-rate', type=float, default=None,
                    help='mean spurious detections per camera per frame')
    ap.add_argument('--dropout',  type=float, default=None,
                    help='probability a real detection is dropped each frame')
    args = ap.parse_args()

    cameras, used_fallback = projection.load_cameras_or_sim(args.cameras)
    if used_fallback:
        print("[eval] WARNING: configured cameras can't see floor; using built-in sim cameras.")

    scn = scene.load_scenario(args.scenario)
    print(f"[eval] scenario '{scn.get('name', args.scenario)}'")
    overrides = {'duration': args.duration, 'noise': args.noise, 'seed': args.seed,
                 'clutter_rate': args.clutter_rate, 'dropout': args.dropout}
    stats = scene.generate(scn, cameras, args.out, overrides)
    print(f"[eval] generated {stats['n_frames']} frames, {stats['n_dets']} dets "
          f"(+{stats['n_clutter']} clutter) -> {stats['recording']}")

    out_jsonl = args.out + '_replay_out.jsonl'

    # Choose runner.
    local = find_local_replay(args.replay)
    runner = args.runner
    if runner == 'auto':
        runner = 'local' if local else ('docker' if docker_available() else None)
    if runner == 'local' and not local:
        print("[eval] ERROR: local runner requested but no binary found "
              "(build it or pass --replay).", file=sys.stderr)
        return 2
    if runner == 'docker' and not docker_available():
        print("[eval] ERROR: docker runner requested but docker is unavailable.", file=sys.stderr)
        return 2
    if runner is None:
        print("[eval] ERROR: no local binary and no docker. Build heimdall_replay "
              "(see tools/README.md) or start Docker.", file=sys.stderr)
        return 2

    if runner == 'local':
        rc = run_local(local, stats['recording'], args.cameras, out_jsonl, args.bypass_tracker)
    else:
        rc = run_docker(stats['recording'], args.cameras, out_jsonl, args.bypass_tracker)
    if rc != 0:
        print(f"[eval] heimdall_replay ({runner}) exited {rc}", file=sys.stderr)
        return rc

    metrics_json = args.out + '_metrics.json'
    mcmd = [sys.executable, os.path.join(HERE, 'replay_metrics.py'),
            out_jsonl, stats['ground_truth'],
            '--recording', stats['recording'], '--cameras', args.cameras,
            '--gate', str(args.gate), '--json', metrics_json]
    return subprocess.run(mcmd, cwd=REPO).returncode


if __name__ == '__main__':
    raise SystemExit(main())
