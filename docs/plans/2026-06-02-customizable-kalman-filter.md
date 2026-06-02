# Customizable Kalman Filter Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CP, CV (default), and CA Kalman filter models selectable at startup, with CA exposing ax/ay through the full C++→proto→Java stack.

**Architecture:** `Track` holds a max-size state (`float[6]`) and covariance (`float[36]`) plus a `FilterModel` enum field. Per-model named functions (`kalman_predict_cp/cv/ca`, `kalman_update_cp/cv/ca`) use their own stride constant; dispatcher wrappers (`kalman_predict`, `kalman_update_combined`) route by `track.model` — JPDA requires no changes. ax/ay flow via proto fields 8/9 and an updated Java client.

**Tech Stack:** C++20, CMake 3.22+, protobuf v26.1, Catch2 v3.5.3, Java (hand-rolled protobuf, no external deps)

**Assumptions:**
- Default `filter_model = CONSTANT_VELOCITY` — no behavior change unless config is set. Will NOT work with runtime model switching (tracks would have mismatched state layout).
- Single `PROCESS_NOISE_Q` tuning knob shared across all models; per-model tuning is out of scope.
- Build directory may or may not exist; `cmake -B build -S .` is idempotent.

---

## File Structure

| File | Action |
|------|--------|
| `src/tracker/track.h` | Extend struct: FilterModel enum, state→`float[6]`, cov→`float[36]`, add `model` field |
| `src/tracker/kalman.h` | New per-model declarations; make_track gains FilterModel param |
| `src/tracker/kalman.cpp` | Rename CV funcs; CP/CA stubs→implementations; dispatchers; updated make_track |
| `src/tracker/tracker.h` | Add `filter_model` to `ObjectTrackerConfig` |
| `src/tracker/tracker.cpp` | Pass model to make_track; ax/ay in to_tracked_object |
| `src/tracker/track_event.h` | Add `ax`, `ay` to `TrackedObject` |
| `src/comm/comm_layer.cpp` | Serialize proto fields 8/9 |
| `proto/heimdall.proto` | Add `float ax = 8; float ay = 9;` to TrackedObjectMsg |
| `tests/test_kalman.cpp` | Fix make_track calls; add CP and CA test sections |
| `tests/test_jpda.cpp` | Fix make_track calls |
| `vendordep/src/main/java/com/heimdall/TrackedObject.java` | Add ax, ay fields + getters |
| `vendordep/src/main/java/com/heimdall/proto/ProtoReader.java` | Parse fields 8, 9 |

---

### Task 1: Extend Track struct and add FilterModel enum

**Files:**
- Modify: `src/tracker/track.h`

**Security flag:** `none`

**Does NOT cover:** Kalman function bodies — track.h change alone will break compilation of kalman.cpp until Task 2.

- [ ] **Step 1: Replace track.h**

```cpp
#pragma once
#include <array>
#include <cstdint>

enum class TrackStatus { TENTATIVE, CONFIRMED };

enum class FilterModel {
    CONSTANT_POSITION,      // state = [x, y],                N=2
    CONSTANT_VELOCITY,      // state = [x, y, vx, vy],        N=4
    CONSTANT_ACCELERATION,  // state = [x, y, vx, vy, ax, ay] N=6
};

// Internal tracker state. Not exposed to callers — use TrackEvent / TrackedObject instead.
struct Track {
    uint32_t    id;
    int         class_id;
    float       confidence;
    FilterModel model;

    // state[0..N-1] active; rest zero-initialised. N = 2, 4, or 6 per model.
    std::array<float, 6>  state;
    // cov uses row-major stride N×N occupying first N*N elements; rest zero.
    std::array<float, 36> cov;

    double      last_update_s;
    int         frames_seen;
    int         frames_missed;
    TrackStatus status;
};
```

- [ ] **Step 2: Verify (compilation will fail at kalman.cpp — expected until Task 2)**

```bash
cmake -B build -S . 2>&1 | tail -5
cmake --build build --target heimdall_tracker 2>&1 | grep "error:" | head -10
```
Expected: errors in kalman.cpp mentioning `std::array<float, 16>` copy-init. These are fixed in Task 2.

- [ ] **Step 3: Commit**

```bash
git add src/tracker/track.h
git commit -m "feat(tracker): extend Track state/cov to float[6]/float[36], add FilterModel enum"
```

---

### Task 2: Update kalman.h and kalman.cpp — rename CV, stubs for CP/CA, dispatchers, new make_track

**Files:**
- Modify: `src/tracker/kalman.h`
- Modify: `src/tracker/kalman.cpp`

**Security flag:** `none`

**Does NOT cover:** CP/CA full implementations (stubs only — tests for those added in Tasks 4/6, implemented in Tasks 5/7).

- [ ] **Step 1: Replace kalman.h**

```cpp
#pragma once
#include "track.h"

// Measurement noise variance (m^2). Derived from +/-7 cm pose accuracy: 0.07^2 ~= 0.005
inline constexpr float MEAS_NOISE_R    = 0.005f;
// Process noise intensity. Higher = trust detections more, allow faster acceleration.
inline constexpr float PROCESS_NOISE_Q = 0.1f;

// Per-model predict (each uses its own stride N = 2, 4, or 6).
void kalman_predict_cp(Track& track, double dt);
void kalman_predict_cv(Track& track, double dt);
void kalman_predict_ca(Track& track, double dt);

// Per-model update with combined JPDAF innovation.
void kalman_update_cp(Track& track, float innov_x, float innov_y, float total_weight);
void kalman_update_cv(Track& track, float innov_x, float innov_y, float total_weight);
void kalman_update_ca(Track& track, float innov_x, float innov_y, float total_weight);

// Dispatchers — called by jpda.cpp; route to the correct model function via track.model.
void kalman_predict(Track& track, double dt);
void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight);

// Construct a new tentative track with the given filter model.
Track make_track(uint32_t id, int class_id, float x, float y,
                 float confidence, double timestamp_s, FilterModel model);
```

- [ ] **Step 2: Replace kalman.cpp**

```cpp
#include "kalman.h"
#include <algorithm>
#include <cmath>

// ─── Constant-Position (CP) — stubs replaced by real impl in Task 5 ─────────

void kalman_predict_cp(Track& /*track*/, double /*dt*/) {}

void kalman_update_cp(Track& /*track*/, float /*innov_x*/, float /*innov_y*/, float /*w*/) {}

// ─── Constant-Velocity (CV) — renamed from original kalman_predict/update ───

void kalman_predict_cv(Track& track, double dt_d) {
    constexpr int N = 4;
    const float dt = static_cast<float>(dt_d);
    auto& x = track.state;
    auto& P = track.cov;

    x[0] += x[2] * dt;
    x[1] += x[3] * dt;

    std::array<float, N*N> FP;
    std::copy_n(P.begin(), N*N, FP.begin());
    for (int j = 0; j < N; ++j) {
        FP[0*N+j] = P[0*N+j] + dt * P[2*N+j];
        FP[1*N+j] = P[1*N+j] + dt * P[3*N+j];
    }
    for (int i = 0; i < N; ++i) {
        P[i*N+0] = FP[i*N+0] + dt * FP[i*N+2];
        P[i*N+1] = FP[i*N+1] + dt * FP[i*N+3];
        P[i*N+2] = FP[i*N+2];
        P[i*N+3] = FP[i*N+3];
    }
    const float q   = PROCESS_NOISE_Q;
    const float dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
    P[0*N+0] += q * dt4 / 4.f;
    P[1*N+1] += q * dt4 / 4.f;
    P[2*N+2] += q * dt2;
    P[3*N+3] += q * dt2;
    P[0*N+2] += q * dt3 / 2.f;   P[2*N+0] += q * dt3 / 2.f;
    P[1*N+3] += q * dt3 / 2.f;   P[3*N+1] += q * dt3 / 2.f;
}

void kalman_update_cv(Track& track, float innov_x, float innov_y, float total_weight) {
    if (total_weight <= 0.f) return;
    constexpr int N = 4;
    auto& x = track.state;
    auto& P = track.cov;

    const float R   = MEAS_NOISE_R;
    const float s00 = P[0*N+0] + R,  s01 = P[0*N+1];
    const float s10 = P[1*N+0],       s11 = P[1*N+1] + R;
    const float det = s00 * s11 - s01 * s10;

    const float si00 =  s11 / det,  si01 = -s01 / det;
    const float si10 = -s10 / det,  si11 =  s00 / det;

    float K[N][2];
    for (int i = 0; i < N; ++i) {
        K[i][0] = P[i*N+0] * si00 + P[i*N+1] * si10;
        K[i][1] = P[i*N+0] * si01 + P[i*N+1] * si11;
    }

    for (int i = 0; i < N; ++i)
        x[i] += K[i][0] * innov_x + K[i][1] * innov_y;

    const float p0[N] = { P[0*N+0], P[0*N+1], P[0*N+2], P[0*N+3] };
    const float p1[N] = { P[1*N+0], P[1*N+1], P[1*N+2], P[1*N+3] };
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            P[i*N+j] -= total_weight * (K[i][0] * p0[j] + K[i][1] * p1[j]);
}

// ─── Constant-Acceleration (CA) — stubs replaced by real impl in Task 7 ─────

void kalman_predict_ca(Track& /*track*/, double /*dt*/) {}

void kalman_update_ca(Track& /*track*/, float /*innov_x*/, float /*innov_y*/, float /*w*/) {}

// ─── Dispatchers ─────────────────────────────────────────────────────────────

void kalman_predict(Track& track, double dt) {
    switch (track.model) {
        case FilterModel::CONSTANT_POSITION:     kalman_predict_cp(track, dt); break;
        case FilterModel::CONSTANT_VELOCITY:     kalman_predict_cv(track, dt); break;
        case FilterModel::CONSTANT_ACCELERATION: kalman_predict_ca(track, dt); break;
    }
}

void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight) {
    switch (track.model) {
        case FilterModel::CONSTANT_POSITION:     kalman_update_cp(track, innov_x, innov_y, total_weight); break;
        case FilterModel::CONSTANT_VELOCITY:     kalman_update_cv(track, innov_x, innov_y, total_weight); break;
        case FilterModel::CONSTANT_ACCELERATION: kalman_update_ca(track, innov_x, innov_y, total_weight); break;
    }
}

// ─── Track construction ───────────────────────────────────────────────────────

Track make_track(uint32_t id, int class_id, float x, float y, float confidence,
                 double timestamp_s, FilterModel model) {
    Track t{};
    t.id            = id;
    t.class_id      = class_id;
    t.confidence    = confidence;
    t.model         = model;
    t.state         = {};
    t.state[0]      = x;
    t.state[1]      = y;
    t.cov           = {};

    switch (model) {
        case FilterModel::CONSTANT_POSITION:
            t.cov[0*2+0] = 10.f;
            t.cov[1*2+1] = 10.f;
            break;
        case FilterModel::CONSTANT_VELOCITY:
            t.cov[0*4+0] = 10.f;
            t.cov[1*4+1] = 10.f;
            t.cov[2*4+2] = 1.f;
            t.cov[3*4+3] = 1.f;
            break;
        case FilterModel::CONSTANT_ACCELERATION:
            t.cov[0*6+0] = 10.f;
            t.cov[1*6+1] = 10.f;
            t.cov[2*6+2] = 1.f;
            t.cov[3*6+3] = 1.f;
            t.cov[4*6+4] = 0.1f;
            t.cov[5*6+5] = 0.1f;
            break;
    }

    t.last_update_s = timestamp_s;
    t.frames_seen   = 1;
    t.frames_missed = 0;
    t.status        = TrackStatus::TENTATIVE;
    return t;
}
```

- [ ] **Step 3: Build tracker library — must succeed**

```bash
cmake --build build --target heimdall_tracker 2>&1 | tail -10
```
Expected: build succeeds with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/tracker/kalman.h src/tracker/kalman.cpp
git commit -m "feat(kalman): rename CV functions, add CP/CA stubs, dispatchers, update make_track"
```

---

### Task 3: Fix make_track callers in existing tests

**Files:**
- Modify: `tests/test_kalman.cpp`
- Modify: `tests/test_jpda.cpp`

**Security flag:** `none`

**Does NOT cover:** New CP/CA tests — those are added in Tasks 4 and 6.

- [ ] **Step 1: Replace tests/test_kalman.cpp** (add `FilterModel::CONSTANT_VELOCITY` to every `make_track` call)

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tracker/kalman.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("predict with dt=0 does not change position", "[kalman]") {
    Track t = make_track(1, 0, 3.f, 4.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    t.state[2] = 1.f; t.state[3] = -1.f;
    kalman_predict(t, 0.0);
    REQUIRE_THAT(t.state[0], WithinAbs(3.f, 1e-5f));
    REQUIRE_THAT(t.state[1], WithinAbs(4.f, 1e-5f));
}

TEST_CASE("predict moves position by velocity * dt", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    t.state[2] = 2.f;
    t.state[3] = -1.f;
    kalman_predict(t, 0.5);
    REQUIRE_THAT(t.state[0], WithinAbs(1.f,  1e-4f));
    REQUIRE_THAT(t.state[1], WithinAbs(-0.5f, 1e-4f));
}

TEST_CASE("predict increases position variance", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    float p00_before = t.cov[0];
    kalman_predict(t, 0.02);
    REQUIRE(t.cov[0] > p00_before);
}

TEST_CASE("update with weight=0 is a no-op", "[kalman]") {
    Track t = make_track(1, 0, 2.f, 3.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    auto state_before = t.state;
    auto cov_before   = t.cov;
    kalman_update_combined(t, 100.f, 100.f, 0.f);
    REQUIRE(t.state == state_before);
    REQUIRE(t.cov   == cov_before);
}

TEST_CASE("update reduces position variance", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    float p00_before = t.cov[0];
    kalman_update_combined(t, 1.f, 1.f, 1.f);
    REQUIRE(t.cov[0] < p00_before);
}

TEST_CASE("update pulls state toward measurement", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    t.cov = {100.f, 0.f, 0.f,  0.f,
             0.f,  100.f, 0.f, 0.f,
             0.f,  0.f,  100.f, 0.f,
             0.f,  0.f,  0.f,  100.f};
    kalman_update_combined(t, 5.f, 3.f, 1.f);
    REQUIRE_THAT(t.state[0], WithinAbs(5.f, 0.1f));
    REQUIRE_THAT(t.state[1], WithinAbs(3.f, 0.1f));
}

TEST_CASE("make_track initializes at given position with zero velocity", "[kalman]") {
    Track t = make_track(42, 3, 1.5f, -2.f, 0.8f, 1.23, FilterModel::CONSTANT_VELOCITY);
    REQUIRE(t.id          == 42);
    REQUIRE(t.class_id    == 3);
    REQUIRE_THAT(t.state[0], WithinAbs(1.5f, 1e-6f));
    REQUIRE_THAT(t.state[1], WithinAbs(-2.f, 1e-6f));
    REQUIRE_THAT(t.state[2], WithinAbs(0.f,  1e-6f));
    REQUIRE_THAT(t.state[3], WithinAbs(0.f,  1e-6f));
    REQUIRE(t.status        == TrackStatus::TENTATIVE);
    REQUIRE(t.frames_seen   == 1);
    REQUIRE(t.frames_missed == 0);
}
```

- [ ] **Step 2: Replace tests/test_jpda.cpp** (add `FilterModel::CONSTANT_VELOCITY` to every `make_track` call)

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tracker/jpda.h"
#include "tracker/kalman.h"

using Catch::Matchers::WithinAbs;

static JpdaConfig default_cfg() {
    return {.gate_distance = 1.0f, .clutter_density = 1.0f, .p_detection = 0.9f};
}

TEST_CASE("empty tracks -- all detections returned as unassociated", "[jpda]") {
    std::vector<Track> tracks;
    std::vector<FieldDetection> dets = {{0, 1.f, 2.f, 0.9f}, {0, 5.f, 5.f, 0.8f}};
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.size() == 2);
    REQUIRE(unassoc[0] == 0);
    REQUIRE(unassoc[1] == 1);
}

TEST_CASE("empty detections -- all tracks get frames_missed incremented", "[jpda]") {
    std::vector<Track> tracks = {
        make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY)
    };
    jpda_update(tracks, {}, 0.02, default_cfg());
    REQUIRE(tracks[0].frames_missed == 1);
    REQUIRE(tracks[0].frames_seen   == 1);
}

TEST_CASE("nearby detection is associated -- not returned as unassociated", "[jpda]") {
    std::vector<Track> tracks = {
        make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY)
    };
    std::vector<FieldDetection> dets = {{0, 0.1f, 0.1f, 0.9f}};
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.empty());
}

TEST_CASE("far detection is unassociated -- outside gate", "[jpda]") {
    std::vector<Track> tracks = {
        make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY)
    };
    std::vector<FieldDetection> dets = {{0, 5.f, 5.f, 0.9f}};
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.size() == 1);
    REQUIRE(unassoc[0] == 0);
}

TEST_CASE("associated detection updates track state toward measurement", "[jpda]") {
    std::vector<Track> tracks = {
        make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY)
    };
    float meas_x = 0.5f, meas_y = 0.3f;
    std::vector<FieldDetection> dets = {{0, meas_x, meas_y, 0.9f}};
    float x_before = tracks[0].state[0];
    jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(std::abs(tracks[0].state[0] - meas_x) < std::abs(x_before - meas_x));
}

TEST_CASE("associated detection increments frames_seen, resets frames_missed", "[jpda]") {
    std::vector<Track> tracks = {
        make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY)
    };
    tracks[0].frames_missed = 2;
    std::vector<FieldDetection> dets = {{0, 0.1f, 0.f, 0.9f}};
    jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(tracks[0].frames_seen   == 2);
    REQUIRE(tracks[0].frames_missed == 0);
}

TEST_CASE("two well-separated tracks -- each associated with its own detection", "[jpda]") {
    std::vector<Track> tracks = {
        make_track(1, 0,  0.f,  0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY),
        make_track(2, 0, 10.f, 10.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY),
    };
    std::vector<FieldDetection> dets = {
        {0, 0.1f,  0.1f,  0.9f},
        {0, 10.1f, 10.1f, 0.9f},
    };
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.empty());
    REQUIRE(tracks[0].frames_missed == 0);
    REQUIRE(tracks[1].frames_missed == 0);
}
```

- [ ] **Step 3: Build and run tracker tests — all existing tests must pass**

```bash
cmake --build build --target heimdall_tracker_tests 2>&1 | tail -5
ctest --test-dir build -R heimdall_tracker_tests --output-on-failure
```
Expected: all tests PASS (same count as before — 6 kalman + 7 jpda + 8 tracker).

- [ ] **Step 4: Commit**

```bash
git add tests/test_kalman.cpp tests/test_jpda.cpp
git commit -m "test(kalman/jpda): update make_track calls with explicit FilterModel::CONSTANT_VELOCITY"
```

---

### Task 4: Write failing CP tests

**Files:**
- Modify: `tests/test_kalman.cpp`

**Security flag:** `none`

**Does NOT cover:** CP implementation — these tests must FAIL with the current stubs.

- [ ] **Step 1: Append CP tests to tests/test_kalman.cpp**

Add at the end of the file:

```cpp
// ─── Constant-Position (CP) tests ─────────────────────────────────────────

TEST_CASE("CP: predict does not change position", "[kalman][cp]") {
    Track t = make_track(1, 0, 3.f, 4.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    kalman_predict(t, 0.1);
    REQUIRE_THAT(t.state[0], WithinAbs(3.f, 1e-5f));
    REQUIRE_THAT(t.state[1], WithinAbs(4.f, 1e-5f));
}

TEST_CASE("CP: predict increases position variance", "[kalman][cp]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    float p00_before = t.cov[0*2+0];
    kalman_predict(t, 0.02);
    REQUIRE(t.cov[0*2+0] > p00_before);
}

TEST_CASE("CP: update reduces position variance", "[kalman][cp]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    float p00_before = t.cov[0*2+0];
    kalman_update_combined(t, 1.f, 1.f, 1.f);
    REQUIRE(t.cov[0*2+0] < p00_before);
}

TEST_CASE("CP: update pulls state toward measurement", "[kalman][cp]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    t.cov[0*2+0] = 100.f;
    t.cov[1*2+1] = 100.f;
    kalman_update_combined(t, 5.f, 3.f, 1.f);
    REQUIRE_THAT(t.state[0], WithinAbs(5.f, 0.1f));
    REQUIRE_THAT(t.state[1], WithinAbs(3.f, 0.1f));
}

TEST_CASE("CP: update with weight=0 is a no-op", "[kalman][cp]") {
    Track t = make_track(1, 0, 2.f, 3.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    auto state_before = t.state;
    auto cov_before   = t.cov;
    kalman_update_combined(t, 100.f, 100.f, 0.f);
    REQUIRE(t.state == state_before);
    REQUIRE(t.cov   == cov_before);
}

TEST_CASE("CP: make_track initializes with zero velocity slots", "[kalman][cp]") {
    Track t = make_track(1, 0, 2.f, -1.f, 0.9f, 0.5, FilterModel::CONSTANT_POSITION);
    REQUIRE(t.model == FilterModel::CONSTANT_POSITION);
    REQUIRE_THAT(t.state[0], WithinAbs(2.f,  1e-6f));
    REQUIRE_THAT(t.state[1], WithinAbs(-1.f, 1e-6f));
    REQUIRE_THAT(t.state[2], WithinAbs(0.f,  1e-6f));  // unused vx
    REQUIRE_THAT(t.state[3], WithinAbs(0.f,  1e-6f));  // unused vy
    REQUIRE_THAT(t.cov[0*2+0], WithinAbs(10.f, 1e-5f));
    REQUIRE_THAT(t.cov[1*2+1], WithinAbs(10.f, 1e-5f));
    REQUIRE_THAT(t.cov[1*2+0], WithinAbs(0.f,  1e-5f));  // off-diagonal zero
}
```

- [ ] **Step 2: Build and run — CP variance/update tests must FAIL**

```bash
cmake --build build --target heimdall_tracker_tests 2>&1 | tail -5
ctest --test-dir build -R "heimdall_tracker_tests" --output-on-failure 2>&1 | grep -E "FAIL|PASS|cp"
```
Expected: `CP: predict increases position variance` FAIL, `CP: update reduces position variance` FAIL, `CP: update pulls state toward measurement` FAIL. The no-op and init tests PASS.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_kalman.cpp
git commit -m "test(kalman): add failing CP model tests"
```

---

### Task 5: Implement CP predict and update

**Files:**
- Modify: `src/tracker/kalman.cpp`

**Security flag:** `none`

**Does NOT cover:** CA model — that is Task 7.

- [ ] **Step 1: Replace CP stubs in kalman.cpp with real implementations**

Replace the two stub bodies (lines starting with `void kalman_predict_cp` and `void kalman_update_cp`) with:

```cpp
void kalman_predict_cp(Track& track, double dt_d) {
    constexpr int N = 2;
    const float dt = static_cast<float>(dt_d);
    auto& P = track.cov;
    // F = I2 — position unchanged (random walk)
    // P = F*P*F' + Q = P + q*dt*I2
    const float q = PROCESS_NOISE_Q;
    P[0*N+0] += q * dt;
    P[1*N+1] += q * dt;
}

void kalman_update_cp(Track& track, float innov_x, float innov_y, float total_weight) {
    if (total_weight <= 0.f) return;
    constexpr int N = 2;
    auto& x = track.state;
    auto& P = track.cov;

    const float R   = MEAS_NOISE_R;
    const float s00 = P[0*N+0] + R,  s01 = P[0*N+1];
    const float s10 = P[1*N+0],       s11 = P[1*N+1] + R;
    const float det = s00 * s11 - s01 * s10;

    const float si00 =  s11 / det,  si01 = -s01 / det;
    const float si10 = -s10 / det,  si11 =  s00 / det;

    float K[N][2];
    for (int i = 0; i < N; ++i) {
        K[i][0] = P[i*N+0] * si00 + P[i*N+1] * si10;
        K[i][1] = P[i*N+0] * si01 + P[i*N+1] * si11;
    }

    for (int i = 0; i < N; ++i)
        x[i] += K[i][0] * innov_x + K[i][1] * innov_y;

    float p0[N], p1[N];
    for (int j = 0; j < N; ++j) { p0[j] = P[0*N+j]; p1[j] = P[1*N+j]; }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            P[i*N+j] -= total_weight * (K[i][0] * p0[j] + K[i][1] * p1[j]);
}
```

- [ ] **Step 2: Build and run — all CP tests must pass, existing tests must still pass**

```bash
cmake --build build --target heimdall_tracker_tests 2>&1 | tail -5
ctest --test-dir build -R heimdall_tracker_tests --output-on-failure
```
Expected: all tests PASS (including all CP tests).

- [ ] **Step 3: Commit**

```bash
git add src/tracker/kalman.cpp
git commit -m "feat(kalman): implement constant-position (CP) predict and update"
```

---

### Task 6: Write failing CA tests

**Files:**
- Modify: `tests/test_kalman.cpp`

**Security flag:** `none`

**Does NOT cover:** CA implementation — these tests must FAIL with the current stubs.

- [ ] **Step 1: Append CA tests to tests/test_kalman.cpp**

```cpp
// ─── Constant-Acceleration (CA) tests ─────────────────────────────────────

TEST_CASE("CA: predict advances position with velocity and acceleration", "[kalman][ca]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    t.state[2] = 1.f;  // vx = 1 m/s
    t.state[4] = 2.f;  // ax = 2 m/s^2
    kalman_predict(t, 1.0);
    // x = 0 + 1*1 + 0.5*2*1^2 = 2.0
    REQUIRE_THAT(t.state[0], WithinAbs(2.f,  1e-4f));
    // vx = 1 + 2*1 = 3.0
    REQUIRE_THAT(t.state[2], WithinAbs(3.f,  1e-4f));
    // ax unchanged
    REQUIRE_THAT(t.state[4], WithinAbs(2.f,  1e-4f));
}

TEST_CASE("CA: predict increases position variance", "[kalman][ca]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    float p00_before = t.cov[0*6+0];
    kalman_predict(t, 0.02);
    REQUIRE(t.cov[0*6+0] > p00_before);
}

TEST_CASE("CA: update reduces position variance", "[kalman][ca]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    float p00_before = t.cov[0*6+0];
    kalman_update_combined(t, 1.f, 1.f, 1.f);
    REQUIRE(t.cov[0*6+0] < p00_before);
}

TEST_CASE("CA: update pulls position toward measurement", "[kalman][ca]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    t.cov[0*6+0] = 100.f;
    t.cov[1*6+1] = 100.f;
    kalman_update_combined(t, 5.f, 3.f, 1.f);
    REQUIRE_THAT(t.state[0], WithinAbs(5.f, 0.1f));
    REQUIRE_THAT(t.state[1], WithinAbs(3.f, 0.1f));
}

TEST_CASE("CA: update with weight=0 is a no-op", "[kalman][ca]") {
    Track t = make_track(1, 0, 2.f, 3.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    auto state_before = t.state;
    auto cov_before   = t.cov;
    kalman_update_combined(t, 100.f, 100.f, 0.f);
    REQUIRE(t.state == state_before);
    REQUIRE(t.cov   == cov_before);
}

TEST_CASE("CA: make_track initializes with zero vel/accel", "[kalman][ca]") {
    Track t = make_track(1, 0, 1.f, 2.f, 0.9f, 0.5, FilterModel::CONSTANT_ACCELERATION);
    REQUIRE(t.model == FilterModel::CONSTANT_ACCELERATION);
    REQUIRE_THAT(t.state[0], WithinAbs(1.f, 1e-6f));
    REQUIRE_THAT(t.state[1], WithinAbs(2.f, 1e-6f));
    for (int i = 2; i < 6; ++i)
        REQUIRE_THAT(t.state[i], WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT(t.cov[0*6+0], WithinAbs(10.f,  1e-5f));
    REQUIRE_THAT(t.cov[1*6+1], WithinAbs(10.f,  1e-5f));
    REQUIRE_THAT(t.cov[2*6+2], WithinAbs(1.f,   1e-5f));
    REQUIRE_THAT(t.cov[3*6+3], WithinAbs(1.f,   1e-5f));
    REQUIRE_THAT(t.cov[4*6+4], WithinAbs(0.1f,  1e-5f));
    REQUIRE_THAT(t.cov[5*6+5], WithinAbs(0.1f,  1e-5f));
}
```

- [ ] **Step 2: Build and run — CA dynamics/variance/update tests must FAIL**

```bash
cmake --build build --target heimdall_tracker_tests 2>&1 | tail -5
ctest --test-dir build -R heimdall_tracker_tests --output-on-failure 2>&1 | grep -E "FAIL|ca\]"
```
Expected: `CA: predict advances position`, `CA: predict increases position variance`, `CA: update reduces position variance`, `CA: update pulls position` all FAIL. Init and no-op tests PASS.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_kalman.cpp
git commit -m "test(kalman): add failing CA model tests"
```

---

### Task 7: Implement CA predict and update

**Files:**
- Modify: `src/tracker/kalman.cpp`

**Security flag:** `none`

**Does NOT cover:** Tracker or wire-format changes — those are Tasks 8–10.

- [ ] **Step 1: Replace CA stubs in kalman.cpp with real implementations**

Replace the two stub bodies (`void kalman_predict_ca` and `void kalman_update_ca`) with:

```cpp
void kalman_predict_ca(Track& track, double dt_d) {
    constexpr int N = 6;
    const float dt  = static_cast<float>(dt_d);
    const float dt2 = dt * dt;
    const float hdt2 = 0.5f * dt2;
    auto& x = track.state;
    auto& P = track.cov;

    // State: x' = F*x
    x[0] += x[2] * dt + x[4] * hdt2;
    x[1] += x[3] * dt + x[5] * hdt2;
    x[2] += x[4] * dt;
    x[3] += x[5] * dt;
    // x[4], x[5] (ax, ay) unchanged

    // P = F*P*F' + Q
    // Step 1: FP = F*P
    std::array<float, N*N> FP;
    std::copy_n(P.begin(), N*N, FP.begin());
    for (int j = 0; j < N; ++j) {
        FP[0*N+j] = P[0*N+j] + dt * P[2*N+j] + hdt2 * P[4*N+j];
        FP[1*N+j] = P[1*N+j] + dt * P[3*N+j] + hdt2 * P[5*N+j];
        FP[2*N+j] = P[2*N+j] + dt * P[4*N+j];
        FP[3*N+j] = P[3*N+j] + dt * P[5*N+j];
        // rows 4, 5 already copied (identity rows of F)
    }
    // Step 2: P = FP*F'
    for (int i = 0; i < N; ++i) {
        P[i*N+0] = FP[i*N+0] + dt * FP[i*N+2] + hdt2 * FP[i*N+4];
        P[i*N+1] = FP[i*N+1] + dt * FP[i*N+3] + hdt2 * FP[i*N+5];
        P[i*N+2] = FP[i*N+2] + dt * FP[i*N+4];
        P[i*N+3] = FP[i*N+3] + dt * FP[i*N+5];
        P[i*N+4] = FP[i*N+4];
        P[i*N+5] = FP[i*N+5];
    }
    // Step 3: P += Q (discrete white-noise-jerk, x/y decoupled)
    const float q   = PROCESS_NOISE_Q;
    const float dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
    P[0*N+0] += q * dt5 / 20.f;
    P[1*N+1] += q * dt5 / 20.f;
    P[0*N+2] += q * dt4 / 8.f;   P[2*N+0] += q * dt4 / 8.f;
    P[1*N+3] += q * dt4 / 8.f;   P[3*N+1] += q * dt4 / 8.f;
    P[0*N+4] += q * dt3 / 6.f;   P[4*N+0] += q * dt3 / 6.f;
    P[1*N+5] += q * dt3 / 6.f;   P[5*N+1] += q * dt3 / 6.f;
    P[2*N+2] += q * dt3 / 3.f;
    P[3*N+3] += q * dt3 / 3.f;
    P[2*N+4] += q * dt2 / 2.f;   P[4*N+2] += q * dt2 / 2.f;
    P[3*N+5] += q * dt2 / 2.f;   P[5*N+3] += q * dt2 / 2.f;
    P[4*N+4] += q * dt;
    P[5*N+5] += q * dt;
}

void kalman_update_ca(Track& track, float innov_x, float innov_y, float total_weight) {
    if (total_weight <= 0.f) return;
    constexpr int N = 6;
    auto& x = track.state;
    auto& P = track.cov;

    const float R   = MEAS_NOISE_R;
    const float s00 = P[0*N+0] + R,  s01 = P[0*N+1];
    const float s10 = P[1*N+0],       s11 = P[1*N+1] + R;
    const float det = s00 * s11 - s01 * s10;

    const float si00 =  s11 / det,  si01 = -s01 / det;
    const float si10 = -s10 / det,  si11 =  s00 / det;

    float K[N][2];
    for (int i = 0; i < N; ++i) {
        K[i][0] = P[i*N+0] * si00 + P[i*N+1] * si10;
        K[i][1] = P[i*N+0] * si01 + P[i*N+1] * si11;
    }

    for (int i = 0; i < N; ++i)
        x[i] += K[i][0] * innov_x + K[i][1] * innov_y;

    float p0[N], p1[N];
    for (int j = 0; j < N; ++j) { p0[j] = P[0*N+j]; p1[j] = P[1*N+j]; }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            P[i*N+j] -= total_weight * (K[i][0] * p0[j] + K[i][1] * p1[j]);
}
```

- [ ] **Step 2: Build and run — all tests must pass**

```bash
cmake --build build --target heimdall_tracker_tests 2>&1 | tail -5
ctest --test-dir build -R heimdall_tracker_tests --output-on-failure
```
Expected: ALL tests PASS (6 CV + 6 CP + 6 CA + 7 JPDA + 8 tracker = 33 tests).

- [ ] **Step 3: Commit**

```bash
git add src/tracker/kalman.cpp
git commit -m "feat(kalman): implement constant-acceleration (CA) predict and update"
```

---

### Task 8: Update tracker — config and TrackedObject output

**Files:**
- Modify: `src/tracker/tracker.h`
- Modify: `src/tracker/tracker.cpp`

**Security flag:** `none`

**Does NOT cover:** Wire format or Java client — those are Tasks 9–10. TrackedObject ax/ay are set but not yet transmitted until Task 9.

- [ ] **Step 1: Update tracker.h**

Add `filter_model` field to `ObjectTrackerConfig`:

```cpp
#pragma once
#include "field_detection.h"
#include "jpda.h"
#include "track.h"
#include "track_event.h"
#include <vector>

struct ObjectTrackerConfig {
    int         confirmation_frames = 3;
    int         loss_frames         = 5;
    float       gate_distance       = 1.0f;
    float       clutter_density     = 1.0f;
    float       p_detection         = 0.9f;
    FilterModel filter_model        = FilterModel::CONSTANT_VELOCITY;
};

class ObjectTracker {
public:
    using Config = ObjectTrackerConfig;

    explicit ObjectTracker(Config config = {});

    std::vector<TrackEvent> update(
        const std::vector<FieldDetection>& detections,
        double timestamp_s
    );

private:
    Config             config_;
    std::vector<Track> tracks_;
    uint32_t           next_id_ = 1;

    TrackedObject to_tracked_object(const Track& t) const;
};
```

- [ ] **Step 2: Update tracker.cpp**

Replace `to_tracked_object` and the `make_track` call inside `update`:

```cpp
#include "tracker.h"
#include "kalman.h"
#include <algorithm>

ObjectTracker::ObjectTracker(Config config) : config_(config) {}

TrackedObject ObjectTracker::to_tracked_object(const Track& t) const {
    const float vx = (t.model != FilterModel::CONSTANT_POSITION) ? t.state[2] : 0.f;
    const float vy = (t.model != FilterModel::CONSTANT_POSITION) ? t.state[3] : 0.f;
    const float ax = (t.model == FilterModel::CONSTANT_ACCELERATION) ? t.state[4] : 0.f;
    const float ay = (t.model == FilterModel::CONSTANT_ACCELERATION) ? t.state[5] : 0.f;
    return {
        .track_id   = t.id,
        .class_id   = t.class_id,
        .x          = t.state[0],
        .y          = t.state[1],
        .vx         = vx,
        .vy         = vy,
        .ax         = ax,
        .ay         = ay,
        .confidence = t.confidence,
    };
}

std::vector<TrackEvent> ObjectTracker::update(
    const std::vector<FieldDetection>& detections,
    double timestamp_s
) {
    JpdaConfig jpda_cfg{
        .gate_distance   = config_.gate_distance,
        .clutter_density = config_.clutter_density,
        .p_detection     = config_.p_detection,
    };

    auto unassociated = jpda_update(tracks_, detections, timestamp_s, jpda_cfg);

    for (int idx : unassociated) {
        const auto& d = detections[idx];
        tracks_.push_back(make_track(next_id_++, d.class_id, d.x, d.y, d.confidence,
                                     timestamp_s, config_.filter_model));
    }

    std::vector<TrackEvent> events;
    std::vector<Track> surviving;

    for (auto& t : tracks_) {
        if (t.status == TrackStatus::TENTATIVE) {
            if (t.frames_missed > 0) {
                continue;
            }
            if (t.frames_seen >= config_.confirmation_frames) {
                t.status = TrackStatus::CONFIRMED;
                events.push_back({TrackEventType::CONFIRMED, to_tracked_object(t)});
            }
            surviving.push_back(std::move(t));
            continue;
        }

        if (t.frames_missed >= config_.loss_frames) {
            events.push_back({TrackEventType::LOST, to_tracked_object(t)});
            continue;
        }

        events.push_back({TrackEventType::UPDATED, to_tracked_object(t)});
        surviving.push_back(std::move(t));
    }

    tracks_ = std::move(surviving);
    return events;
}
```

Note: `to_tracked_object` references `TrackedObject::ax` and `TrackedObject::ay` which don't exist yet in `track_event.h`. This file will fail to compile until Task 9, Step 1 adds those fields. Proceed to Task 9 immediately.

- [ ] **Step 3: Commit (will not compile until Task 9 Step 1 — commit as WIP)**

```bash
git add src/tracker/tracker.h src/tracker/tracker.cpp
git commit -m "feat(tracker): add filter_model config; expose ax/ay in to_tracked_object"
```

---

### Task 9: Add ax/ay to TrackedObject, proto, and comm

**Files:**
- Modify: `src/tracker/track_event.h`
- Modify: `proto/heimdall.proto`
- Modify: `src/comm/comm_layer.cpp`

**Security flag:** `none`

**Does NOT cover:** Java client — that is Task 10. Proto3 additive fields 8/9 are backward compatible: old clients ignore them via `default: skipField()`.

- [ ] **Step 1: Update track_event.h** — add ax/ay fields

```cpp
#pragma once
#include <cstdint>

enum class TrackEventType {
    CONFIRMED,
    UPDATED,
    LOST,
};

struct TrackedObject {
    uint32_t track_id;
    int      class_id;
    float    x, y;
    float    vx, vy;
    float    ax, ay;     // 0.f when model != CONSTANT_ACCELERATION
    float    confidence;
};

struct TrackEvent {
    TrackEventType type;
    TrackedObject  object;
};
```

- [ ] **Step 2: Update proto/heimdall.proto** — add fields 8 and 9

```proto
syntax = "proto3";
package heimdall;

message RobotPoseMsg {
    float  x            = 1;
    float  y            = 2;
    float  heading      = 3;
    uint64 timestamp_ns = 4;
}

message TrackedObjectMsg {
    uint32 track_id   = 1;
    int32  class_id   = 2;
    float  x          = 3;
    float  y          = 4;
    float  vx         = 5;
    float  vy         = 6;
    float  confidence = 7;
    float  ax         = 8;
    float  ay         = 9;
}

enum TrackEventTypeMsg {
    CONFIRMED = 0;
    UPDATED   = 1;
    LOST      = 2;
}

message TrackEventMsg {
    TrackEventTypeMsg type   = 1;
    TrackedObjectMsg  object = 2;
}

message DetectionFrameMsg {
    repeated TrackEventMsg events       = 1;
    uint64                 timestamp_ns = 2;
    bool                   healthy      = 3;
}

message RawDetectionMsg {
    int32  camera_id  = 1;
    int32  class_id   = 2;
    float  confidence = 3;
    float  left       = 4;
    float  top        = 5;
    float  width      = 6;
    float  height     = 7;
}

message RawDetectionFrameMsg {
    repeated RawDetectionMsg detections  = 1;
    uint64                   timestamp_ns = 2;
}
```

- [ ] **Step 3: Update comm_layer.cpp** — serialize ax/ay

After `obj->set_vy(ev.object.vy);` add:
```cpp
        obj->set_ax(ev.object.ax);
        obj->set_ay(ev.object.ay);
```

The full updated block in `send_frame`:
```cpp
    for (const auto& ev : events) {
        auto* msg_ev = frame.add_events();
        switch (ev.type) {
            case TrackEventType::CONFIRMED: msg_ev->set_type(heimdall::CONFIRMED); break;
            case TrackEventType::UPDATED:   msg_ev->set_type(heimdall::UPDATED);   break;
            case TrackEventType::LOST:      msg_ev->set_type(heimdall::LOST);      break;
            default: break;
        }
        auto* obj = msg_ev->mutable_object();
        obj->set_track_id(ev.object.track_id);
        obj->set_class_id(ev.object.class_id);
        obj->set_x(ev.object.x);
        obj->set_y(ev.object.y);
        obj->set_vx(ev.object.vx);
        obj->set_vy(ev.object.vy);
        obj->set_ax(ev.object.ax);
        obj->set_ay(ev.object.ay);
        obj->set_confidence(ev.object.confidence);
    }
```

- [ ] **Step 4: Build and run all tracker tests**

```bash
cmake --build build --target heimdall_tracker_tests 2>&1 | tail -5
ctest --test-dir build -R heimdall_tracker_tests --output-on-failure
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tracker/track_event.h proto/heimdall.proto src/comm/comm_layer.cpp
git commit -m "feat(wire): add ax/ay to TrackedObject, proto fields 8/9, comm serialization"
```

---

### Task 10: Update Java vendordep

**Files:**
- Modify: `vendordep/src/main/java/com/heimdall/TrackedObject.java`
- Modify: `vendordep/src/main/java/com/heimdall/proto/ProtoReader.java`

**Security flag:** `none`

**Does NOT cover:** Java tests (none exist). Manual verification: parse a DetectionFrame and call `getAx()`/`getAy()`.

- [ ] **Step 1: Replace TrackedObject.java**

```java
package com.heimdall;

/** Field-relative tracked object with Kalman-filtered pose, velocity, and acceleration. */
public final class TrackedObject {
    private final int trackId;
    private final int classId;
    private final double x;
    private final double y;
    private final double vx;
    private final double vy;
    private final double ax;   // 0.0 when filter model is not CONSTANT_ACCELERATION
    private final double ay;
    private final double confidence;

    public TrackedObject(int trackId, int classId, double x, double y,
                         double vx, double vy, double ax, double ay, double confidence) {
        this.trackId    = trackId;
        this.classId    = classId;
        this.x          = x;
        this.y          = y;
        this.vx         = vx;
        this.vy         = vy;
        this.ax         = ax;
        this.ay         = ay;
        this.confidence = confidence;
    }

    public int    getTrackId()    { return trackId; }
    public int    getClassId()    { return classId; }
    public double getX()          { return x; }
    public double getY()          { return y; }
    public double getVx()         { return vx; }
    public double getVy()         { return vy; }
    public double getAx()         { return ax; }
    public double getAy()         { return ay; }
    public double getConfidence() { return confidence; }

    @Override
    public String toString() {
        return String.format(
            "TrackedObject{id=%d cls=%d pos=(%.2f,%.2f) vel=(%.2f,%.2f) acc=(%.2f,%.2f) conf=%.2f}",
            trackId, classId, x, y, vx, vy, ax, ay, confidence);
    }
}
```

- [ ] **Step 2: Update ProtoReader.java**

Replace the `parseTrackedObject` method:

```java
    private static TrackedObject parseTrackedObject(byte[] data) {
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int trackId = 0, classId = 0;
        float x = 0, y = 0, vx = 0, vy = 0, ax = 0, ay = 0, confidence = 0;

        while (buf.hasRemaining()) {
            long tag = readVarint(buf);
            int field    = (int)(tag >>> 3);
            int wireType = (int)(tag & 0x7);

            switch (field) {
                case 1: trackId    = (int) readVarint(buf); break;
                case 2: classId    = (int) readVarint(buf); break;
                case 3: x          = buf.getFloat();         break;
                case 4: y          = buf.getFloat();         break;
                case 5: vx         = buf.getFloat();         break;
                case 6: vy         = buf.getFloat();         break;
                case 7: confidence = buf.getFloat();         break;
                case 8: ax         = buf.getFloat();         break;
                case 9: ay         = buf.getFloat();         break;
                default: skipField(buf, wireType);           break;
            }
        }
        return new TrackedObject(trackId, classId, x, y, vx, vy, ax, ay, confidence);
    }
```

Also update the default `TrackedObject` initialization in `parseTrackEvent` from 7-arg to 9-arg:

```java
        TrackedObject obj = new TrackedObject(0, 0, 0, 0, 0, 0, 0, 0, 0);
```

- [ ] **Step 3: Build the vendordep (Gradle)**

```bash
cd vendordep && ./gradlew build 2>&1 | tail -10
```
Expected: BUILD SUCCESSFUL.

- [ ] **Step 4: Commit**

```bash
cd ..
git add vendordep/src/main/java/com/heimdall/TrackedObject.java
git add vendordep/src/main/java/com/heimdall/proto/ProtoReader.java
git commit -m "feat(vendordep): add ax/ay to TrackedObject and ProtoReader (proto fields 8/9)"
```

---

## Self-Review

**Spec coverage:**
- ✅ CP model: Tasks 1–5
- ✅ CV model (default): Task 2 (rename, preserved behavior), default in Task 8
- ✅ CA model: Tasks 1–2, 6–7
- ✅ ax/ay in TrackedObject: Task 9 (track_event.h) + Task 8 (to_tracked_object)
- ✅ Proto fields 8/9: Task 9
- ✅ comm_layer serialization: Task 9
- ✅ Java TrackedObject: Task 10
- ✅ Java ProtoReader: Task 10
- ✅ FilterModel in ObjectTrackerConfig: Task 8
- ✅ make_track passes model to Track: Task 2 + Task 8

**No placeholders present.**

**Type consistency:**
- `FilterModel` defined in `track.h` (Task 1), used in kalman.h/cpp (Task 2), tracker.h/cpp (Task 8)
- `TrackedObject::ax/ay` added in Task 9 Step 1, referenced in Task 8 Step 2 — Task 8 note says compile order matters; Task 9 must follow immediately
- `make_track` signature in kalman.h matches implementation in kalman.cpp throughout

**Scope:** No "basic" or "v1" language. All models fully implemented.
