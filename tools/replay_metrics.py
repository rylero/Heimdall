"""
replay_metrics.py -- score heimdall_replay output against ground truth.

Consumes the REAL pipeline output (heimdall_replay's output.jsonl, produced by
the actual DetectionProcessor / JPDAF / pose projection) and the sim's
ground-truth file, and reports objective quality numbers you can regress against:

  - position error vs range   (mean / 95p, and binned by distance from origin)
  - ID switches               (a GT object whose matched track_id changes over time)
  - false tracks              (track events matched to no GT object)
  - recall / track dropouts   (GT objects that SHOULD be visible but aren't tracked)
  - confirm latency           (time from an object first becoming visible to first track)

Visibility gating (recommended): pass --recording (and --cameras) so the tool can
re-project each GT object through the SAME camera model that generated the
detections (tools/sim/projection.py) using the recorded robot poses. Then "missed"
and "recall" only count objects actually in a camera's field of view -- ground
truth lists every object every frame, including ones off-camera, so without this
those counts are dominated by out-of-view objects and are not meaningful.

Without --recording, the tool falls back to gating "missed" to objects the
pipeline tracks at least once, and prints a caveat.

Usage:
  python tools/replay_metrics.py <output.jsonl> <ground_truth.jsonl> \
      [--recording <recording.jsonl>] [--cameras config/cameras] \
      [--gate 0.75] [--json out_metrics.json]

Schemas (authoritative, from the C++):
  output line: {"ts_ns","healthy","events":[{"type","track_id","class_id","x","y",...}]}
  gt line:     {"ts_ns","objects":[{"obj_id","class_id","x","y","vx","vy"}]}
  rec pose:    {"t":"pose","x","y","heading","ts_ns",...}
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import projection  # noqa: E402


def load_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values, p):
    if not values:
        return float('nan')
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def greedy_match(events, gt_objects, gate):
    """Greedily match track events to GT objects (same class, nearest within gate).
    Returns (pairs, unmatched_events, unmatched_gt), pairs = [(event, gt_obj, dist)]."""
    candidates = []
    for ei, ev in enumerate(events):
        for gi, go in enumerate(gt_objects):
            if int(ev.get('class_id', -1)) != int(go.get('class_id', -2)):
                continue
            d = math.hypot(float(ev['x']) - float(go['x']), float(ev['y']) - float(go['y']))
            if d <= gate:
                candidates.append((d, ei, gi))
    candidates.sort(key=lambda c: c[0])
    used_e, used_g, pairs = set(), set(), []
    for d, ei, gi in candidates:
        if ei in used_e or gi in used_g:
            continue
        used_e.add(ei);  used_g.add(gi)
        pairs.append((events[ei], gt_objects[gi], d))
    unmatched_e = [ev for ei, ev in enumerate(events) if ei not in used_e]
    unmatched_g = [go for gi, go in enumerate(gt_objects) if gi not in used_g]
    return pairs, unmatched_e, unmatched_g


def load_poses(recording_path):
    """Return a sorted list of (ts_ns, x, y, heading) from a recording's pose lines."""
    poses = []
    for r in load_jsonl(recording_path):
        if r.get('t') == 'pose':
            poses.append((int(r['ts_ns']), float(r['x']), float(r['y']), float(r['heading'])))
    poses.sort(key=lambda p: p[0])
    return poses


def nearest_pose(poses, ts):
    """Nearest recorded pose to ts_ns (poses and frames run at different rates)."""
    if not poses:
        return None
    import bisect
    ks = [p[0] for p in poses]
    i = bisect.bisect_left(ks, ts)
    best = poses[min(i, len(poses) - 1)]
    if i > 0 and abs(poses[i - 1][0] - ts) < abs(best[0] - ts):
        best = poses[i - 1]
    return best


def is_visible(cameras, pose, go):
    """True if GT object go is projected into any camera's frame at this robot pose."""
    _, rx, ry, heading = pose
    for cam in cameras:
        if projection.field_to_pixel(cam, rx, ry, heading, float(go['x']), float(go['y'])) is not None:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Score heimdall_replay output vs ground truth")
    ap.add_argument('output', help='heimdall_replay output.jsonl')
    ap.add_argument('ground_truth', help='sim <out>_ground_truth.jsonl')
    ap.add_argument('--recording', default=None,
                    help='sim <out>_recording.jsonl -- enables visibility-gated recall/missed')
    ap.add_argument('--cameras', default='config/cameras',
                    help='camera config dir (used only with --recording)')
    ap.add_argument('--gate', type=float, default=0.75,
                    help='max match distance (m) between a track and a GT object')
    ap.add_argument('--json', default=None, help='write metrics as JSON to this path')
    args = ap.parse_args()

    out_rows = load_jsonl(args.output)
    gt_rows  = load_jsonl(args.ground_truth)
    gt_by_ts = {int(r['ts_ns']): r.get('objects', []) for r in gt_rows}

    # Optional visibility gating via the shared camera model + recorded poses.
    poses, cameras, gated = None, None, False
    if args.recording:
        poses = load_poses(args.recording)
        cameras, _ = projection.load_cameras_or_sim(args.cameras)
        gated = True

    pos_errors, err_by_range = [], {}
    gt_to_track, id_switches = {}, 0
    gt_first_visible, confirm_latency_ns = {}, {}
    offgate_track_frames = 0           # active-track frames whose estimate is > gate from any GT
    all_track_ids = set()              # every track_id the pipeline emitted
    matched_track_ids = set()          # track_ids that matched a real GT object at least once
    total_track_events = matched_events = 0
    visible_object_frames = 0          # GT objects in view (gated) or all GT (ungated)
    missed_object_frames = 0
    ever_matched = set()
    frames_scored = frames_unaligned = 0

    for row in out_rows:
        ts = int(row['ts_ns'])
        events = [e for e in row.get('events', []) if 'track_id' in e and 'x' in e]
        total_track_events += len(events)

        gt_objects = gt_by_ts.get(ts)
        if gt_objects is None:
            frames_unaligned += 1
            continue
        frames_scored += 1

        # Determine which GT objects are in view this frame.
        if gated:
            pose = nearest_pose(poses, ts)
            visible = [go for go in gt_objects if pose and is_visible(cameras, pose, go)]
        else:
            visible = gt_objects  # no visibility info; treat all as "expected"

        for e in events:
            all_track_ids.add(int(e['track_id']))

        pairs, unmatched_e, unmatched_g = greedy_match(events, gt_objects, args.gate)
        matched_events += len(pairs)
        # An unmatched active-track frame is NOT necessarily a ghost: it is often a
        # real track whose estimate drifted past the gate (range offset / lag).
        # Count these separately; true false positives are tracks that NEVER match.
        offgate_track_frames += len(unmatched_e)
        for ev in pairs:
            matched_track_ids.add(int(ev[0]['track_id']))

        visible_ids = {int(go['obj_id']) for go in visible}
        visible_object_frames += len(visible_ids)
        for go in unmatched_g:
            oid = int(go['obj_id'])
            if gated:
                if oid in visible_ids:
                    missed_object_frames += 1
            # ungated missed is computed after the loop (needs ever_matched)

        # record first-visible time (gated) for confirm latency
        if gated:
            for oid in visible_ids:
                gt_first_visible.setdefault(oid, ts)

        for ev, go, d in pairs:
            oid = int(go['obj_id']);  tid = int(ev['track_id'])
            ever_matched.add(oid)
            pos_errors.append(d)
            rng = math.hypot(float(go['x']), float(go['y']))
            err_by_range.setdefault(round(rng), []).append(d)
            if oid in gt_to_track and gt_to_track[oid] != tid:
                id_switches += 1
            gt_to_track[oid] = tid
            if oid not in confirm_latency_ns:
                first = gt_first_visible.get(oid, ts) if gated else ts
                confirm_latency_ns[oid] = max(0, ts - first)

    if not gated:
        # Fall back: only count dropouts of objects tracked at least once.
        unmatched_per_frame = []
        for row in out_rows:
            ts = int(row['ts_ns'])
            gt_objects = gt_by_ts.get(ts)
            if gt_objects is None:
                continue
            events = [e for e in row.get('events', []) if 'track_id' in e and 'x' in e]
            _, _, unmatched_g = greedy_match(events, gt_objects, args.gate)
            for go in unmatched_g:
                unmatched_per_frame.append(int(go['obj_id']))
        missed_object_frames = sum(1 for oid in unmatched_per_frame if oid in ever_matched)
        visible_object_frames = None  # not meaningful without visibility

    recall = (matched_events / visible_object_frames) if visible_object_frames else float('nan')
    ghost_tracks = sorted(all_track_ids - matched_track_ids)

    metrics = {
        'gated_by_visibility':  gated,
        'frames_scored':        frames_scored,
        'frames_unaligned':     frames_unaligned,
        'tracked_objects':      len(ever_matched),
        'total_tracks':         len(all_track_ids),
        'pos_err_mean_m':       (sum(pos_errors) / len(pos_errors)) if pos_errors else float('nan'),
        'pos_err_p95_m':        percentile(pos_errors, 95),
        'pos_err_max_m':        max(pos_errors) if pos_errors else float('nan'),
        'id_switches':          id_switches,
        'confirm_latency_ms_mean': (
            (sum(confirm_latency_ns.values()) / len(confirm_latency_ns) / 1e6)
            if confirm_latency_ns else float('nan')),
        'ghost_tracks':         len(ghost_tracks),
        'offgate_track_frames': offgate_track_frames,
        'missed_object_frames': missed_object_frames,
        'visible_object_frames': visible_object_frames,
        'recall':               recall,
        'total_track_events':   total_track_events,
        'matched_events':       matched_events,
        'match_rate':           (matched_events / total_track_events) if total_track_events else float('nan'),
        'pos_err_by_range_m':   {str(k): (sum(v) / len(v)) for k, v in sorted(err_by_range.items())},
    }

    def fmt(x):
        return 'n/a' if (isinstance(x, float) and math.isnan(x)) or x is None else (
            f"{x:.3f}" if isinstance(x, float) else str(x))

    print("=" * 52)
    print(f" replay metrics: {args.output}")
    print(f"          vs GT: {args.ground_truth}")
    print(f"      visibility: {'gated (re-projected)' if gated else 'UNGATED (pass --recording to gate)'}")
    print("=" * 52)
    print(f"  frames scored / unaligned : {frames_scored} / {frames_unaligned}")
    print(f"  position error  mean/p95/max (m): "
          f"{fmt(metrics['pos_err_mean_m'])} / {fmt(metrics['pos_err_p95_m'])} / {fmt(metrics['pos_err_max_m'])}")
    print(f"  ID switches               : {metrics['id_switches']}")
    print(f"  confirm latency mean (ms) : {fmt(metrics['confirm_latency_ms_mean'])}")
    print(f"  match rate                : {fmt(metrics['match_rate'])} "
          f"({metrics['matched_events']}/{metrics['total_track_events']})")
    if gated:
        print(f"  recall (visible objects)  : {fmt(metrics['recall'])} "
              f"({metrics['matched_events']}/{metrics['visible_object_frames']})")
    print(f"  tracks total / tracked obj: {metrics['total_tracks']} / {metrics['tracked_objects']}")
    print(f"  ghost tracks (never match): {metrics['ghost_tracks']}   <- true false positives")
    print(f"  off-gate track-frames     : {metrics['offgate_track_frames']}  "
          f"(real tracks drifted > {args.gate} m; mostly range offset, not ghosts)")
    print(f"  track dropouts (frames)   : {metrics['missed_object_frames']}"
          f"{'' if gated else '  (gated to tracked objects)'}")
    if metrics['pos_err_by_range_m']:
        print("  position error by range (m from origin):")
        for rng, err in metrics['pos_err_by_range_m'].items():
            print(f"      ~{rng} m : {err:.3f}")
    print("=" * 52)

    if frames_unaligned and not frames_scored:
        print("  WARNING: no frames aligned by ts_ns -- output and GT are from "
              "different runs. Regenerate both from the same scenario run.")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  metrics json -> {args.json}")


if __name__ == '__main__':
    main()
