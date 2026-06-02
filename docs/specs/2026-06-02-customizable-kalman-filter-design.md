# Customizable Kalman Filter Models

**Date:** 2026-06-02  
**Status:** Approved

## Scope

Add three selectable motion models to the Heimdall tracker:

| Model | Enum | State | Dim |
|-------|------|-------|-----|
| Constant Position | `CONSTANT_POSITION` | [x, y] | 2 |
| Constant Velocity | `CONSTANT_VELOCITY` | [x, y, vx, vy] | 4 |
| Constant Acceleration | `CONSTANT_ACCELERATION` | [x, y, vx, vy, ax, ay] | 6 |

Selected at startup via `ObjectTrackerConfig::filter_model`. Default: `CONSTANT_VELOCITY` (preserves current behavior).

CA exposes `ax`/`ay` through the full stack: C++ `TrackedObject` → proto fields 8/9 → `comm_layer` serialization → Java `TrackedObject` + `ProtoReader`.

**Non-goals:**
- Runtime model switching (would require flushing all tracks)
- Per-track heterogeneous models
- Separate Q-noise tuning knobs per model (single `PROCESS_NOISE_Q` reused)

## Architecture

### Approach: Fixed Max-Size State in Track

`Track` carries `std::array<float, 6>` for state and `std::array<float, 36>` for covariance regardless of model. The active elements are `state[0..N-1]` and `cov[0..N*N-1]` where N = 2, 4, or 6. Unused slots are zero-initialized at track creation.

Each model has its own named predict/update functions using their own stride constant. A dispatcher in `kalman.h` routes based on `track.model`.

Rationale: matches existing free-function style, no heap allocation, no template propagation, minimal blast radius.

### FilterModel Enum (track.h)

```cpp
enum class FilterModel {
    CONSTANT_POSITION,      // N=2
    CONSTANT_VELOCITY,      // N=4  (current behavior)
    CONSTANT_ACCELERATION,  // N=6
};
```

### Track Struct Changes (track.h)

```cpp
struct Track {
    uint32_t   id;
    int        class_id;
    float      confidence;
    FilterModel model;                   // NEW

    std::array<float, 6>  state;         // extended from 4
    std::array<float, 36> cov;           // extended from 16

    double      last_update_s;
    int         frames_seen;
    int         frames_missed;
    TrackStatus status;
};
```

### Kalman Functions (kalman.h / kalman.cpp)

Per-model predict and update:
```cpp
void kalman_predict_cp(Track& track, double dt);   // 2-state, 2x2 cov
void kalman_predict_cv(Track& track, double dt);   // 4-state, 4x4 cov (current logic)
void kalman_predict_ca(Track& track, double dt);   // 6-state, 6x6 cov

void kalman_update_cp(Track& track, float innov_x, float innov_y, float total_weight);
void kalman_update_cv(Track& track, float innov_x, float innov_y, float total_weight);
void kalman_update_ca(Track& track, float innov_x, float innov_y, float total_weight);
```

Dispatcher wrappers (called by jpda.cpp — no JPDA changes needed):
```cpp
void kalman_predict(Track& track, double dt);
void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight);
```

`make_track` gains a `FilterModel` parameter:
```cpp
Track make_track(uint32_t id, int class_id, float x, float y,
                 float confidence, double timestamp_s, FilterModel model);
```

### Motion Models

**CP — Constant Position (random walk)**
- F = I₂
- Q = q·dt·I₂
- P_init = diag(10, 10)

**CV — Constant Velocity (unchanged)**
- F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
- Q = discrete white noise acceleration (current DWNA formula)
- P_init = diag(10, 10, 1, 1)

**CA — Constant Acceleration (new)**
- F (6×6): positions couple to velocity (dt) and acceleration (dt²/2); velocities couple to acceleration (dt)
- Q = discrete white noise jerk model:
  - Q[x,x]=q·dt⁵/20, Q[x,vx]=q·dt⁴/8, Q[x,ax]=q·dt³/6
  - Q[vx,vx]=q·dt³/3, Q[vx,ax]=q·dt²/2
  - Q[ax,ax]=q·dt (and symmetric for y components)
- P_init = diag(10, 10, 1, 1, 0.1, 0.1)

All models: H = [I₂ | 0…] (2×N), measurement = [x, y]. The 2×2 innovation covariance S and Kalman gain derivation are identical in structure — only K is Nx2 instead of 4x2.

### Config Plumbing

```cpp
struct ObjectTrackerConfig {
    int         confirmation_frames = 3;
    int         loss_frames         = 5;
    float       gate_distance       = 1.0f;
    float       clutter_density     = 1.0f;
    float       p_detection         = 0.9f;
    FilterModel filter_model        = FilterModel::CONSTANT_VELOCITY;  // NEW
};
```

`ObjectTracker` stores `filter_model_` and passes it to `make_track`. `jpda.cpp` is unchanged (dispatchers handle routing transparently).

### TrackedObject Changes (track_event.h)

```cpp
struct TrackedObject {
    uint32_t track_id;
    int      class_id;
    float    x, y;
    float    vx, vy;
    float    ax, ay;     // NEW — 0.f unless model == CONSTANT_ACCELERATION
    float    confidence;
};
```

`to_tracked_object` in `tracker.cpp`:
- vx/vy = state[2/3] if model != CP, else 0.f
- ax/ay = state[4/5] if model == CA, else 0.f

### Wire Protocol Changes (proto/heimdall.proto)

```proto
message TrackedObjectMsg {
    uint32 track_id   = 1;
    int32  class_id   = 2;
    float  x          = 3;
    float  y          = 4;
    float  vx         = 5;
    float  vy         = 6;
    float  confidence = 7;
    float  ax         = 8;   // NEW
    float  ay         = 9;   // NEW
}
```

Backward compatible: proto3 skips zero-value fields on the wire. Old Java readers hit `default: skipField()` for unknown fields 8/9. New Java reading old Jetson: fields absent → ax=ay=0.f.

### comm_layer.cpp

Add `obj->set_ax(ev.object.ax)` and `obj->set_ay(ev.object.ay)` after the existing `set_vy` call.

### Java vendordep Changes

**TrackedObject.java** — add `ax`, `ay` fields, constructor param, and getters. `toString()` updated.

**ProtoReader.java** — add `case 8: ax = buf.getFloat(); break;` and `case 9: ay = buf.getFloat(); break;` in `parseTrackedObject`.

## Files Changed

| File | Type |
|------|------|
| `src/tracker/track.h` | Extend struct |
| `src/tracker/kalman.h` | New declarations |
| `src/tracker/kalman.cpp` | Implement CP + CA, keep CV, add dispatchers |
| `src/tracker/tracker.h` | Add `filter_model` to config |
| `src/tracker/tracker.cpp` | Pass model to make_track; update to_tracked_object |
| `src/tracker/track_event.h` | Add ax, ay to TrackedObject |
| `src/comm/comm_layer.cpp` | Serialize ax, ay |
| `proto/heimdall.proto` | Add fields 8, 9 |
| `src/main.cpp` | Config unchanged (default CV) |
| `vendordep/.../TrackedObject.java` | Add ax, ay |
| `vendordep/.../ProtoReader.java` | Parse fields 8, 9 |
| `tests/test_kalman.cpp` | Tests for all three models |

## Testing Strategy

- **CP predict**: position unchanged at constant-dt, variance grows linearly with dt
- **CP update**: same structure as current CV tests (just 2-state)
- **CA predict**: position advances as x + vx·dt + ½ax·dt², velocity advances as vx + ax·dt
- **CA update**: state pulls toward measurement, cov reduces
- **Dispatch**: verify CP/CV/CA dispatch calls correct code path
- **Existing CV tests**: pass unchanged (make_track gains FilterModel param, tests updated to pass it)

## Known Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Cov stride bug (wrong `i*N+j` stride per model) | Critical | Each model function has a local `constexpr int N` stride; unit tests verify numerical covariance values |
| CP track reading vx/vy | Minor | make_track zeroes full array; to_tracked_object guards on model |
| Proto zero omission | Non-issue | Proto3 behavior is correct: ax=ay=0 → not sent → Java defaults to 0 |
