# Heimdall tools

Developer tooling for the Heimdall vision pipeline. The headline workflow is the
**detection-level 3D simulation** loop, which lets you develop and regression-test
the vision pipeline (detection → JPDAF tracker → pose projection) with no robot,
no cameras, and no Jetson/DeepStream — it runs on a plain laptop.

## The sim loop

```
scenario.jsonc ─► sim_recording ─► *_recording.jsonl ─► heimdall_replay ─► *_replay_out.jsonl
   (author the        (project via     *_ground_truth.jsonl    (REAL engine:        │
    3D scene)          shared camera                             DetectionProcessor  │
                       model)                                    + JPDAF + pose)     │
                                                                                     ▼
                                                          replay_metrics ◄───────────┘
                                          (pos error vs range, ID switches, confirm
                                           latency, false/missed tracks → *_metrics.json)
```

Everything downstream of the recording is the **real** C++ pipeline. Only the
scene generation and scoring are synthetic. The camera model lives in exactly one
place — `tools/sim/projection.py` — shared by the generator and the visualizer.

### One command

```bash
python tools/sim_eval.py --scenario tools/scenarios/basic.jsonc
```

This generates the recording + ground truth, runs `heimdall_replay`, and prints a
metrics report (also written to `recordings/eval_metrics.json`). It auto-finds the
replay binary under `build/`; pass `--replay <path>` if yours is elsewhere.

### Step by step

```bash
# 1. Generate a recording + ground truth from a scenario
python tools/sim_recording.py --scenario tools/scenarios/basic.jsonc \
    --cameras config/cameras --out recordings/basic

# 2. Run the real pipeline
./build/heimdall_replay recordings/basic_recording.jsonl config/cameras \
    recordings/basic_replay_out.jsonl

# 3. Score the real output against ground truth
python tools/replay_metrics.py recordings/basic_replay_out.jsonl \
    recordings/basic_ground_truth.jsonl --json recordings/basic_metrics.json

# 4. (optional) Eyeball it: animated top-down view
python tools/compare_replay.py recordings/basic_recording.jsonl \
    recordings/basic_ground_truth.jsonl
```

Omitting `--scenario` reproduces the original hardcoded scene (backward compatible).

## Authoring scenarios

Scenarios are `.jsonc` files under `tools/scenarios/` (`//` comments allowed).
See `basic.jsonc` (straight drive, regression baseline) and `curved_path.jsonc`
(waypoint path with turning). Fields:

| key | meaning |
|-----|---------|
| `duration`, `fps`, `pose_hz`, `noise`, `conf`, `seed` | sim params (CLI flags override) |
| `robot.type` | `linear` (`speed`, `heading`) or `waypoints` (`waypoints:[{t,x,y,heading}]`) |
| `objects[]` | fuel pieces: `class_id`, `x`, `y`, optional `vx`, `vy`, `radius` (m) |
| `apriltags[]` | tag placements — **recorded but not evaluated in v1** (see below) |

**AprilTags in v1:** detection-level sim can only synthesize the final vision
*pose* (a noisy robot pose), which tests robot consumption but not Heimdall's tag
image detection / PnP math. That is a video-level concern (future Phase 3), so tags
are carried in the scenario for later reuse but not scored here.

## Building `heimdall_replay` (Windows)

`heimdall_replay` builds unconditionally — no GStreamer/DeepStream/CUDA. Native
MinGW (MSYS2 UCRT64) works:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target heimdall_replay -j
```

WSL is a fallback if the toolchain misbehaves. The first configure/build is slow
(protobuf + libzmq compile from source via FetchContent).

## Deterministic tuning

The sim is **bitwise reproducible**: the scenario seed fixes the noise RNG and a
fixed timestamp base (`base_ns`) keeps absolute times constant, so the same
inputs + same tracker params always produce the same metrics. `heimdall_replay`
accepts the tracker parameters as flags (`--gate-distance`, `--clutter-density`,
`--confirm-frames`, `--loss-frames`, `--meas-noise-r`, `--process-noise-q`,
`--p-detection`, `--pos-cov-floor`, `--filter-model cp|cv|ca`), so tuning is a
reproducible grid search, not guesswork.

> Note: replay uses the `ObjectTrackerConfig` struct defaults unless you pass
> flags — it does **not** read `config/heimdall.jsonc`. Pass your deployed values
> (e.g. `--gate-distance 2.0 --confirm-frames 2 --loss-frames 15`) to reproduce
> production behaviour.

`tools/sim_sweep.py` runs a grid over noise levels and any tracker params:

```bash
# sweep the gate distance at several noise levels
python tools/sim_sweep.py --scenario tools/scenarios/basic.jsonc \
    --noise 0,5,10,20 --sweep gate_distance=0.5,1.0,2.0

# 2-D grid with a fixed override on every cell
python tools/sim_sweep.py --noise 0,10 \
    --sweep clutter_density=0.5,1,2 --sweep confirm_frames=2,3 --set loss_frames=15
```

Results print as a table and are written to CSV (`--csv`). Read the columns
together: `gate_distance` trades **recall** against **ID switches** and position
error — a gate wider than the object spacing lets a track grab a neighbour's
detection and swap identity. `ghost_tracks` (tracks that never match any real
object) is the true false-positive count; `offgate_track_frames` is real tracks
whose estimate drifted past the match gate (mostly the range-dependent
projection offset), *not* false positives.

## Guardrail

`tools/tests/test_model_sync.py` round-trips known field points through the real
binary and asserts the recovered position matches ground truth — this catches
drift between the Python projector (`tools/sim/projection.py`) and the C++
`pose_estimator.cpp`. If it fails, the sim is lying; fix it before trusting metrics.

## Other tools

- `mock_robot` — standalone RoboRIO ZMQ stand-in (C++ target).
- `calibrate.py`, `calibrate_apriltag_camera.py`, `capture.py` — camera calibration/capture.
- `visualizer.py`, `grid_overlay/` — live visualization helpers.
