"""
sim_sweep.py -- deterministic tracker tuning over noise levels and parameters.

The sim is fully deterministic (fixed scenario seed -> identical recording), and
heimdall_replay now accepts tracker parameters as flags, so a parameter sweep is
reproducible: same inputs + same params => same metrics, every run. That makes
tuning a grid search rather than guesswork.

For each noise level a single recording is generated; then every tracker-param
combination is replayed against it and scored. Results print as a table and are
written to CSV.

Examples:
  # sweep the gate distance at several noise levels
  python tools/sim_sweep.py --scenario tools/scenarios/basic.jsonc \
      --noise 0,5,10,20 --sweep gate_distance=0.5,1.0,2.0

  # 2-D grid, with a fixed override applied to every cell
  python tools/sim_sweep.py --noise 0,10 \
      --sweep clutter_density=0.5,1,2 --sweep confirm_frames=2,3 \
      --set loss_frames=15

Tunable params (name -> replay flag):
  confirm_frames loss_frames gate_distance clutter_density p_detection
  meas_noise_r process_noise_q pos_cov_floor filter_model(cp|cv|ca)
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from sim import projection, scene  # noqa: E402
import sim_eval  # runner selection + generation  # noqa: E402

PARAM_FLAG = {
    'confirm_frames':  '--confirm-frames',
    'loss_frames':     '--loss-frames',
    'gate_distance':   '--gate-distance',
    'clutter_density': '--clutter-density',
    'p_detection':     '--p-detection',
    'meas_noise_r':    '--meas-noise-r',
    'process_noise_q': '--process-noise-q',
    'pos_cov_floor':   '--pos-cov-floor',
    'filter_model':    '--filter-model',
}

# Columns pulled from each metrics.json into the results table.
REPORT_COLS = ['pos_err_mean_m', 'pos_err_p95_m', 'id_switches', 'ghost_tracks',
               'offgate_track_frames', 'recall', 'match_rate', 'total_tracks']


def parse_kv_list(items):
    """['gate_distance=0.5,1.0', ...] -> [('gate_distance', ['0.5','1.0']), ...]."""
    out = []
    for it in items or []:
        k, _, v = it.partition('=')
        if k not in PARAM_FLAG:
            raise SystemExit(f"unknown param '{k}'. Known: {', '.join(PARAM_FLAG)}")
        out.append((k, v.split(',')))
    return out


def run_replay(runner, local, rec, cameras, out_jsonl, flags):
    """Run replay (local or docker) with extra tracker flags."""
    if runner == 'local':
        cmd = [local, rec, cameras, out_jsonl] + flags
        return subprocess.run(cmd, cwd=REPO,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    c_rec = sim_eval.to_container_path(rec, 'recordings', '/recordings')
    c_out = sim_eval.to_container_path(out_jsonl, 'recordings', '/recordings')
    c_cam = sim_eval.to_container_path(cameras, 'config', '/app/config')
    cmd = ['docker', 'compose', '-f', sim_eval.COMPOSE_FILE, 'run', '--rm', 'replay',
           c_rec, c_cam, c_out] + flags
    return subprocess.run(cmd, cwd=REPO,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def score(out_jsonl, gt, recording, cameras, gate):
    with tempfile.NamedTemporaryFile('r', suffix='.json', delete=False) as tf:
        mj = tf.name
    subprocess.run([sys.executable, os.path.join(HERE, 'replay_metrics.py'),
                    out_jsonl, gt, '--recording', recording, '--cameras', cameras,
                    '--gate', str(gate), '--json', mj],
                   cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(mj) as f:
        m = json.load(f)
    os.unlink(mj)
    return m


def main():
    ap = argparse.ArgumentParser(description="Deterministic tracker tuning sweep")
    ap.add_argument('--scenario', default='tools/scenarios/basic.jsonc')
    ap.add_argument('--cameras',  default='config/cameras')
    ap.add_argument('--noise',    default='0,5,10,20', help='comma-separated pixel-noise levels')
    ap.add_argument('--sweep',    action='append', default=[],
                    help='param=v1,v2,... (repeatable; cartesian product)')
    ap.add_argument('--set',      action='append', default=[],
                    help='param=value fixed override applied to every cell (repeatable)')
    ap.add_argument('--runner',   choices=['auto', 'local', 'docker'], default='auto')
    ap.add_argument('--replay',   default=None)
    ap.add_argument('--gate',     type=float, default=0.75, help='metrics match gate (m)')
    ap.add_argument('--out',      default='recordings/sweep', help='output dir/prefix')
    ap.add_argument('--csv',      default='recordings/sweep_results.csv')
    args = ap.parse_args()

    noises = [float(n) for n in args.noise.split(',')]
    sweeps = parse_kv_list(args.sweep)
    fixed = {k: v[0] for k, v in parse_kv_list(args.set)}

    # Runner selection (shared with sim_eval).
    local = sim_eval.find_local_replay(args.replay)
    runner = args.runner
    if runner == 'auto':
        runner = 'local' if local else ('docker' if sim_eval.docker_available() else None)
    if runner is None or (runner == 'docker' and not sim_eval.docker_available()) or \
       (runner == 'local' and not local):
        raise SystemExit("no heimdall_replay runner available (build it or start Docker).")

    cameras, _ = projection.load_cameras_or_sim(args.cameras)
    scn = scene.load_scenario(args.scenario)

    combos = [dict(zip([k for k, _ in sweeps], vals))
              for vals in itertools.product(*[v for _, v in sweeps])] or [{}]

    rows = []
    for noise in noises:
        prefix = f"{args.out}_n{noise:g}"
        stats = scene.generate(scn, cameras, prefix, {'noise': noise})
        for combo in combos:
            params = {**fixed, **combo}
            flags = []
            for k, v in params.items():
                flags += [PARAM_FLAG[k], str(v)]
            tag = '_'.join(f"{k}{v}" for k, v in params.items()) or 'default'
            out_jsonl = f"{prefix}_{tag}_out.jsonl"
            rc = run_replay(runner, local, stats['recording'], args.cameras, out_jsonl, flags)
            if rc != 0:
                print(f"  [warn] replay failed (noise={noise}, {params}) rc={rc}", file=sys.stderr)
                continue
            m = score(out_jsonl, stats['ground_truth'], stats['recording'], args.cameras, args.gate)
            row = {'noise': noise, **params, **{c: m.get(c) for c in REPORT_COLS}}
            rows.append(row)
            print(f"  noise={noise:>5g}  {tag:<28}  "
                  f"posErr={m['pos_err_mean_m']:.3f}  idsw={m['id_switches']:>3}  "
                  f"ghosts={m['ghost_tracks']:>2}  recall={m['recall']:.3f}")

    # Write CSV
    if rows:
        cols = ['noise'] + [k for k, _ in sweeps] + list(fixed.keys()) + REPORT_COLS
        seen, ordered = set(), []
        for c in cols:
            if c not in seen:
                seen.add(c);  ordered.append(c)
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in ordered})
        print(f"\n[sweep] {len(rows)} cells -> {args.csv}")


if __name__ == '__main__':
    main()
