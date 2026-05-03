# Heimdall Milestone 3: JPDA Object Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pure C++20 Joint Probabilistic Data Association (JPDA) tracker that takes field-relative object detections, maintains up to 100 simultaneous tracks with Kalman filtering, and emits CONFIRMED / UPDATED / LOST events — fully unit-testable on any machine with no hardware dependency.

**Architecture:** A `heimdall_tracker` CMake library target (no GStreamer/DeepStream dependency) contains three layers: a constant-velocity Kalman filter operating on 2D field coordinates, a JPDAF association step computing marginal association probabilities and combined innovations, and an `ObjectTracker` public class managing track lifecycle (tentative → confirmed → lost). Input is `FieldDetection` structs (field-relative x/y in meters — output of pose estimation in Milestone 2). Output is `TrackEvent` vectors.

**Tech Stack:** C++20, Catch2 v3 (via FetchContent, already in project), CMake — zero additional dependencies.

> **Testing:** All tasks run on any machine. No Jetson, Docker, or hardware required. Build with `cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build --target heimdall_tracker_tests`.

---

## File Structure

```
src/tracker/
  field_detection.h         # FieldDetection input struct (x,y meters + class_id + confidence)
  track.h                   # Track internal state (Kalman state, covariance, lifecycle counters)
  track_event.h             # TrackedObject + TrackEvent output types
  kalman.h                  # Function declarations: kalman_predict, kalman_update_combined, make_track
  kalman.cpp                # Kalman filter implementation (predict + JPDAF combined update)
  jpda.h                    # jpda_update() declaration + JpdaConfig struct
  jpda.cpp                  # JPDAF association: gate → likelihoods → β probabilities → updates
  tracker.h                 # ObjectTracker public class declaration
  tracker.cpp               # ObjectTracker: wraps JPDA, manages IDs and lifecycle events
tests/
  test_kalman.cpp           # Unit tests: predict/update math, zero-weight no-op, variance changes
  test_jpda.cpp             # Unit tests: gating, association probabilities, unassociated returns
  test_tracker.cpp          # Integration tests: confirmation, loss, multi-object, class preservation
CMakeLists.txt              # Modify: add heimdall_tracker library target (no GStreamer dep)
tests/CMakeLists.txt        # Modify: add heimdall_tracker_tests executable
```

---

## Task 1: CMake — Add Tracker Library + Test Target

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

The tracker has no GStreamer or DeepStream dependency. It must build and test on any machine, not just the Jetson. A separate library target achieves this.

- [ ] **Step 1: Write failing build verification**

Try to build the (not-yet-existing) tracker target:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target heimdall_tracker_tests 2>&1 | head -5
```
Expected: error — `heimdall_tracker_tests` target not found.

- [ ] **Step 2: Add heimdall_tracker to CMakeLists.txt**

In `CMakeLists.txt`, add this block after the `FetchContent_MakeAvailable(Catch2)` line and before the GStreamer section:

```cmake
# ---------------------------------------------------------------------------
# Tracker library — no GStreamer/DeepStream dependency, unit-testable anywhere
# ---------------------------------------------------------------------------
add_library(heimdall_tracker
    src/tracker/kalman.cpp
    src/tracker/jpda.cpp
    src/tracker/tracker.cpp
)
target_include_directories(heimdall_tracker PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
```

Also add `heimdall_tracker` as a dependency of `heimdall_pipeline`. In the existing `target_link_libraries(heimdall_pipeline ...)` block, add `heimdall_tracker` under `PUBLIC`:

```cmake
target_link_libraries(heimdall_pipeline
    PUBLIC
        ${GST_LIBRARIES}
        ${NVDS_META_LIB}
        ${NVDSGST_META_LIB}
        heimdall_tracker
)
```

- [ ] **Step 3: Update tests/CMakeLists.txt**

Replace the entire contents of `tests/CMakeLists.txt` with:

```cmake
# Existing pipeline tests (require GStreamer headers — Jetson/Docker only)
add_executable(heimdall_tests
    test_camera_source.cpp
)
target_link_libraries(heimdall_tests
    heimdall_pipeline
    Catch2::Catch2WithMain
)
include(Catch)
catch_discover_tests(heimdall_tests)

# Tracker tests — no hardware dependency, run on any machine
add_executable(heimdall_tracker_tests
    test_kalman.cpp
    test_jpda.cpp
    test_tracker.cpp
)
target_link_libraries(heimdall_tracker_tests
    heimdall_tracker
    Catch2::Catch2WithMain
)
catch_discover_tests(heimdall_tracker_tests)
```

- [ ] **Step 4: Create stub source files so CMake can configure**

Create these stubs (they are replaced in Tasks 3–5):

`src/tracker/kalman.cpp`:
```cpp
// Stub — implemented in Task 3.
```

`src/tracker/jpda.cpp`:
```cpp
// Stub — implemented in Task 4.
```

`src/tracker/tracker.cpp`:
```cpp
// Stub — implemented in Task 5.
```

`tests/test_kalman.cpp`:
```cpp
// Stub — implemented in Task 3.
```

`tests/test_jpda.cpp`:
```cpp
// Stub — implemented in Task 4.
```

`tests/test_tracker.cpp`:
```cpp
// Stub — implemented in Task 5.
```

- [ ] **Step 5: Verify CMake configures (tracker target resolves)**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

Expected: configures without error. GStreamer/DeepStream errors are acceptable if not in Docker — the key check is that `heimdall_tracker` and `heimdall_tracker_tests` targets are listed:

```bash
cmake --build build --target help 2>&1 | grep -i tracker
```

Expected: `heimdall_tracker` and `heimdall_tracker_tests` both appear.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt tests/CMakeLists.txt \
        src/tracker/kalman.cpp src/tracker/jpda.cpp src/tracker/tracker.cpp \
        tests/test_kalman.cpp tests/test_jpda.cpp tests/test_tracker.cpp
git commit -m "feat: add heimdall_tracker CMake target and tracker test executable"
```

---

## Task 2: Data Types

**Files:**
- Create: `src/tracker/field_detection.h`
- Create: `src/tracker/track.h`
- Create: `src/tracker/track_event.h`
- Create: `src/tracker/kalman.h`
- Create: `src/tracker/jpda.h`
- Create: `src/tracker/tracker.h`

All headers. No implementation. No tests (tested implicitly by later tasks).

- [ ] **Step 1: Create src/tracker/field_detection.h**

```cpp
#pragma once

// Output of pose estimation (Milestone 2). Input to the tracker.
struct FieldDetection {
    int   class_id;
    float x, y;         // field-relative, meters (WPILib coordinate system)
    float confidence;
};
```

- [ ] **Step 2: Create src/tracker/track.h**

```cpp
#pragma once
#include <array>
#include <cstdint>

enum class TrackStatus { TENTATIVE, CONFIRMED };

// Internal tracker state. Not exposed to callers — use TrackEvent / TrackedObject instead.
struct Track {
    uint32_t   id;
    int        class_id;
    float      confidence;

    // Kalman state: [x, y, vx, vy]  (meters, m/s, field-relative)
    std::array<float, 4>  state;
    // Covariance: 4×4 row-major
    std::array<float, 16> cov;

    double      last_update_s;
    int         frames_seen;    // consecutive frames with associated detection
    int         frames_missed;  // consecutive frames with no associated detection
    TrackStatus status;
};
```

- [ ] **Step 3: Create src/tracker/track_event.h**

```cpp
#pragma once
#include <cstdint>

enum class TrackEventType {
    CONFIRMED,  // track promoted from tentative to confirmed
    UPDATED,    // confirmed track updated with new position estimate
    LOST,       // confirmed track has not been seen for loss_frames — retired
};

struct TrackedObject {
    uint32_t track_id;
    int      class_id;
    float    x, y;      // field-relative, meters (Kalman-filtered estimate)
    float    vx, vy;    // estimated velocity, m/s
    float    confidence;
};

struct TrackEvent {
    TrackEventType type;
    TrackedObject  object;
};
```

- [ ] **Step 4: Create src/tracker/kalman.h**

```cpp
#pragma once
#include "track.h"

// Measurement noise variance (m²). Derived from ±7 cm pose accuracy: 0.07² ≈ 0.005
static constexpr float MEAS_NOISE_R    = 0.005f;
// Process noise intensity. Higher = trust detections more, allow faster acceleration.
static constexpr float PROCESS_NOISE_Q = 0.1f;

// Predict track forward by dt seconds (constant-velocity model).
void kalman_predict(Track& track, double dt);

// Update track with combined JPDAF innovation.
//   innov_x, innov_y = Σ_j β_ij * (z_j - H * x_pred)   (weighted innovation sum)
//   total_weight     = Σ_j β_ij = 1 - β_i0
// No-ops when total_weight == 0 (no associated detections this frame).
void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight);

// Construct a new tentative track at (x,y) with zero velocity.
Track make_track(uint32_t id, int class_id, float x, float y, float confidence, double timestamp_s);
```

- [ ] **Step 5: Create src/tracker/jpda.h**

```cpp
#pragma once
#include "field_detection.h"
#include "track.h"
#include <vector>

struct JpdaConfig {
    float gate_distance   = 1.0f;  // meters, Euclidean gating radius
    float clutter_density = 1.0f;  // λ — expected clutter returns per gated region
    float p_detection     = 0.9f;  // P_D — probability of detecting a present target
};

// Run one JPDAF cycle:
//   1. Predict all tracks to timestamp_s
//   2. Compute Gaussian likelihoods within Euclidean gate
//   3. Compute marginal β association probabilities
//   4. Update each track with combined weighted innovation
//   5. Increment frames_seen / frames_missed counters
//
// Returns indices into `detections` with total association probability < 0.5
// (candidates for new track creation).
std::vector<int> jpda_update(
    std::vector<Track>&                tracks,
    const std::vector<FieldDetection>& detections,
    double                             timestamp_s,
    const JpdaConfig&                  cfg
);
```

- [ ] **Step 6: Create src/tracker/tracker.h**

```cpp
#pragma once
#include "field_detection.h"
#include "jpda.h"
#include "track.h"
#include "track_event.h"
#include <vector>

class ObjectTracker {
public:
    struct Config {
        int   confirmation_frames = 3;    // frames_seen threshold: TENTATIVE → CONFIRMED
        int   loss_frames         = 5;    // frames_missed threshold: CONFIRMED → LOST
        float gate_distance       = 1.0f; // meters
        float clutter_density     = 1.0f; // JPDA λ
        float p_detection         = 0.9f; // JPDA P_D
    };

    explicit ObjectTracker(Config config = {});

    // Process one frame. timestamp_s is monotonic seconds (e.g. buf_pts_ns / 1e9).
    // Returns events for this frame: CONFIRMED for newly confirmed tracks,
    // UPDATED for all active confirmed tracks, LOST for expired tracks.
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

- [ ] **Step 7: Commit**

```bash
git add src/tracker/
git commit -m "feat: tracker data types — FieldDetection, Track, TrackEvent, headers"
```

---

## Task 3: Kalman Filter

**Files:**
- Modify: `src/tracker/kalman.cpp` (replace stub)
- Modify: `tests/test_kalman.cpp` (replace stub)

Constant-velocity 2D Kalman filter. State = [x, y, vx, vy]. All matrix math done with fixed-size arrays — no Eigen.

- [ ] **Step 1: Write failing tests (replace tests/test_kalman.cpp stub)**

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tracker/kalman.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("predict with dt=0 does not change position", "[kalman]") {
    Track t = make_track(1, 0, 3.f, 4.f, 1.f, 0.0);
    t.state[2] = 1.f; t.state[3] = -1.f;
    kalman_predict(t, 0.0);
    REQUIRE_THAT(t.state[0], WithinAbs(3.f, 1e-5f));
    REQUIRE_THAT(t.state[1], WithinAbs(4.f, 1e-5f));
}

TEST_CASE("predict moves position by velocity * dt", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0);
    t.state[2] = 2.f;   // vx = 2 m/s
    t.state[3] = -1.f;  // vy = -1 m/s
    kalman_predict(t, 0.5);
    REQUIRE_THAT(t.state[0], WithinAbs(1.f,  1e-4f));
    REQUIRE_THAT(t.state[1], WithinAbs(-0.5f, 1e-4f));
}

TEST_CASE("predict increases position variance", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0);
    float p00_before = t.cov[0];
    kalman_predict(t, 0.02);
    REQUIRE(t.cov[0] > p00_before);
}

TEST_CASE("update with weight=0 is a no-op", "[kalman]") {
    Track t = make_track(1, 0, 2.f, 3.f, 1.f, 0.0);
    auto state_before = t.state;
    auto cov_before   = t.cov;
    kalman_update_combined(t, 100.f, 100.f, 0.f);
    REQUIRE(t.state == state_before);
    REQUIRE(t.cov   == cov_before);
}

TEST_CASE("update reduces position variance", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0);
    float p00_before = t.cov[0];
    kalman_update_combined(t, 1.f, 1.f, 1.f);
    REQUIRE(t.cov[0] < p00_before);
}

TEST_CASE("update pulls state toward measurement", "[kalman]") {
    // Small covariance — update should strongly pull state toward measurement.
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0);
    t.cov = {0.0001f, 0.f, 0.f,    0.f,
             0.f,    0.0001f, 0.f,  0.f,
             0.f,    0.f,    0.0001f, 0.f,
             0.f,    0.f,    0.f,    0.0001f};
    // Innovation = meas - pred = (5,3) - (0,0)
    kalman_update_combined(t, 5.f, 3.f, 1.f);
    REQUIRE_THAT(t.state[0], WithinAbs(5.f, 0.1f));
    REQUIRE_THAT(t.state[1], WithinAbs(3.f, 0.1f));
}

TEST_CASE("make_track initializes at given position with zero velocity", "[kalman]") {
    Track t = make_track(42, 3, 1.5f, -2.f, 0.8f, 1.23);
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

- [ ] **Step 2: Build to confirm failure**

```bash
cmake --build build --target heimdall_tracker_tests 2>&1 | tail -5
```

Expected: link error or compile error — `kalman_predict` not defined.

- [ ] **Step 3: Implement src/tracker/kalman.cpp (replace stub)**

```cpp
#include "kalman.h"
#include <cmath>

void kalman_predict(Track& track, double dt_d) {
    const float dt = static_cast<float>(dt_d);
    auto& x = track.state;
    auto& P = track.cov;   // row-major: P[i*4+j]

    // Predict state: position += velocity * dt
    x[0] += x[2] * dt;
    x[1] += x[3] * dt;

    // Propagate covariance: P = F*P*F' + Q
    // F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
    //
    // Step 1: FP = F*P  (rows 0,1 of P mix in rows 2,3 via dt)
    std::array<float, 16> FP = P;
    for (int j = 0; j < 4; ++j) {
        FP[0*4+j] = P[0*4+j] + dt * P[2*4+j];
        FP[1*4+j] = P[1*4+j] + dt * P[3*4+j];
    }
    // Step 2: P = FP*F'  (cols 0,1 of FP mix in cols 2,3 via dt)
    for (int i = 0; i < 4; ++i) {
        P[i*4+0] = FP[i*4+0] + dt * FP[i*4+2];
        P[i*4+1] = FP[i*4+1] + dt * FP[i*4+3];
        P[i*4+2] = FP[i*4+2];
        P[i*4+3] = FP[i*4+3];
    }
    // Step 3: P += Q  (discrete white noise acceleration model)
    const float q    = PROCESS_NOISE_Q;
    const float dt2  = dt * dt;
    const float dt3  = dt2 * dt;
    const float dt4  = dt3 * dt;
    P[0*4+0] += q * dt4 / 4.f;
    P[1*4+1] += q * dt4 / 4.f;
    P[2*4+2] += q * dt2;
    P[3*4+3] += q * dt2;
    P[0*4+2] += q * dt3 / 2.f;   P[2*4+0] += q * dt3 / 2.f;
    P[1*4+3] += q * dt3 / 2.f;   P[3*4+1] += q * dt3 / 2.f;
}

void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight) {
    if (total_weight <= 0.f) return;

    auto& x = track.state;
    auto& P = track.cov;

    // S = H*P*H' + R  (2×2; H = [[1,0,0,0],[0,1,0,0]] selects top-left block of P)
    const float R   = MEAS_NOISE_R;
    const float s00 = P[0*4+0] + R,  s01 = P[0*4+1];
    const float s10 = P[1*4+0],       s11 = P[1*4+1] + R;
    const float det = s00 * s11 - s01 * s10;

    // S⁻¹  (analytical 2×2 inverse)
    const float si00 =  s11 / det,  si01 = -s01 / det;
    const float si10 = -s10 / det,  si11 =  s00 / det;

    // K = P*H'*S⁻¹  (4×2)
    // P*H' = first two columns of P
    float K[4][2];
    for (int i = 0; i < 4; ++i) {
        K[i][0] = P[i*4+0] * si00 + P[i*4+1] * si10;
        K[i][1] = P[i*4+0] * si01 + P[i*4+1] * si11;
    }

    // State update: x += K * innov
    x[0] += K[0][0] * innov_x + K[0][1] * innov_y;
    x[1] += K[1][0] * innov_x + K[1][1] * innov_y;
    x[2] += K[2][0] * innov_x + K[2][1] * innov_y;
    x[3] += K[3][0] * innov_x + K[3][1] * innov_y;

    // Covariance update: P = (I - w*K*H) * P
    // Save rows 0,1 before in-place modification (rows 2,3 unchanged by H)
    const float p0[4] = { P[0], P[1], P[2], P[3] };
    const float p1[4] = { P[4], P[5], P[6], P[7] };
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            P[i*4+j] -= total_weight * (K[i][0] * p0[j] + K[i][1] * p1[j]);
}

Track make_track(uint32_t id, int class_id, float x, float y, float confidence, double timestamp_s) {
    Track t{};
    t.id            = id;
    t.class_id      = class_id;
    t.confidence    = confidence;
    t.state         = { x, y, 0.f, 0.f };
    t.cov           = {
        10.f, 0.f,  0.f,  0.f,
        0.f,  10.f, 0.f,  0.f,
        0.f,  0.f,  1.f,  0.f,
        0.f,  0.f,  0.f,  1.f,
    };
    t.last_update_s = timestamp_s;
    t.frames_seen   = 1;
    t.frames_missed = 0;
    t.status        = TrackStatus::TENTATIVE;
    return t;
}
```

- [ ] **Step 4: Build and run tests**

```bash
cmake --build build --target heimdall_tracker_tests
cd build && ctest -R heimdall_tracker_tests --output-on-failure
```

Expected:
```
Test  1: predict with dt=0 does not change position         PASSED
Test  2: predict moves position by velocity * dt            PASSED
Test  3: predict increases position variance                PASSED
Test  4: update with weight=0 is a no-op                   PASSED
Test  5: update reduces position variance                   PASSED
Test  6: update pulls state toward measurement              PASSED
Test  7: make_track initializes at given position ...       PASSED
7 tests passed, 0 failed
```

- [ ] **Step 5: Commit**

```bash
git add src/tracker/kalman.cpp tests/test_kalman.cpp
git commit -m "feat: constant-velocity Kalman filter with JPDAF combined update"
```

---

## Task 4: JPDA Association

**Files:**
- Modify: `src/tracker/jpda.cpp` (replace stub)
- Modify: `tests/test_jpda.cpp` (replace stub)

Marginal JPDAF: Euclidean gating → Gaussian likelihoods → β probabilities → combined innovations → kalman_update_combined.

- [ ] **Step 1: Write failing tests (replace tests/test_jpda.cpp stub)**

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tracker/jpda.h"
#include "tracker/kalman.h"

using Catch::Matchers::WithinAbs;

static JpdaConfig default_cfg() {
    return {.gate_distance = 1.0f, .clutter_density = 1.0f, .p_detection = 0.9f};
}

TEST_CASE("empty tracks — all detections returned as unassociated", "[jpda]") {
    std::vector<Track> tracks;
    std::vector<FieldDetection> dets = {{0, 1.f, 2.f, 0.9f}, {0, 5.f, 5.f, 0.8f}};
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.size() == 2);
    REQUIRE(unassoc[0] == 0);
    REQUIRE(unassoc[1] == 1);
}

TEST_CASE("empty detections — all tracks get frames_missed incremented", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    jpda_update(tracks, {}, 0.02, default_cfg());
    REQUIRE(tracks[0].frames_missed == 1);
    REQUIRE(tracks[0].frames_seen   == 1);
}

TEST_CASE("nearby detection is associated — not returned as unassociated", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    std::vector<FieldDetection> dets = {{0, 0.1f, 0.1f, 0.9f}};  // 0.14m away — inside 1m gate
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.empty());
}

TEST_CASE("far detection is unassociated — outside gate", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    std::vector<FieldDetection> dets = {{0, 5.f, 5.f, 0.9f}};  // 7m away — outside 1m gate
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.size() == 1);
    REQUIRE(unassoc[0] == 0);
}

TEST_CASE("associated detection updates track state toward measurement", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    float meas_x = 0.5f, meas_y = 0.3f;
    std::vector<FieldDetection> dets = {{0, meas_x, meas_y, 0.9f}};
    float x_before = tracks[0].state[0];
    jpda_update(tracks, dets, 0.02, default_cfg());
    // Position should move toward measurement
    REQUIRE(std::abs(tracks[0].state[0] - meas_x) < std::abs(x_before - meas_x));
}

TEST_CASE("associated detection increments frames_seen, resets frames_missed", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    tracks[0].frames_missed = 2;
    std::vector<FieldDetection> dets = {{0, 0.1f, 0.f, 0.9f}};
    jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(tracks[0].frames_seen   == 2);
    REQUIRE(tracks[0].frames_missed == 0);
}

TEST_CASE("two well-separated tracks with nearby detections — each associated correctly", "[jpda]") {
    std::vector<Track> tracks = {
        make_track(1, 0,  0.f,  0.f, 1.f, 0.0),
        make_track(2, 0, 10.f, 10.f, 1.f, 0.0),
    };
    std::vector<FieldDetection> dets = {
        {0, 0.1f,  0.1f,  0.9f},  // near track 1
        {0, 10.1f, 10.1f, 0.9f},  // near track 2
    };
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.empty());
    REQUIRE(tracks[0].frames_missed == 0);
    REQUIRE(tracks[1].frames_missed == 0);
}
```

- [ ] **Step 2: Build to confirm failure**

```bash
cmake --build build --target heimdall_tracker_tests 2>&1 | tail -5
```

Expected: link error — `jpda_update` not defined.

- [ ] **Step 3: Implement src/tracker/jpda.cpp (replace stub)**

```cpp
#include "jpda.h"
#include "kalman.h"
#include <cmath>
#include <vector>

static constexpr float TWO_PI = 6.28318530718f;

std::vector<int> jpda_update(
    std::vector<Track>&                tracks,
    const std::vector<FieldDetection>& detections,
    double                             timestamp_s,
    const JpdaConfig&                  cfg
) {
    const int n = static_cast<int>(tracks.size());
    const int m = static_cast<int>(detections.size());

    // 1. Predict all tracks forward
    for (auto& t : tracks) {
        double dt = timestamp_s - t.last_update_s;
        if (dt > 0.0) kalman_predict(t, dt);
    }

    // Early-out: no tracks or no detections
    if (n == 0 || m == 0) {
        for (auto& t : tracks) {
            ++t.frames_missed;
            t.last_update_s = timestamp_s;
        }
        std::vector<int> all(m);
        for (int j = 0; j < m; ++j) all[j] = j;
        return all;
    }

    // 2. Compute Gaussian likelihoods within Euclidean gate
    //    Uses diagonal S approximation: S ≈ diag(P[0,0]+R, P[1,1]+R)
    std::vector<std::vector<float>> L(n, std::vector<float>(m, 0.f));
    for (int i = 0; i < n; ++i) {
        const auto& P  = tracks[i].cov;
        const float s00 = P[0*4+0] + MEAS_NOISE_R;
        const float s11 = P[1*4+1] + MEAS_NOISE_R;
        const float norm = 1.f / (TWO_PI * std::sqrt(s00 * s11));

        for (int j = 0; j < m; ++j) {
            float dx = detections[j].x - tracks[i].state[0];
            float dy = detections[j].y - tracks[i].state[1];
            if (std::sqrt(dx*dx + dy*dy) >= cfg.gate_distance) continue;
            float maha = (dx*dx) / s00 + (dy*dy) / s11;
            L[i][j] = norm * std::exp(-0.5f * maha);
        }
    }

    // 3. Compute marginal JPDAF association probabilities
    //    β[i][j] = P_D * L[i][j] / (λ + P_D * Σ_k L[i][k])
    std::vector<std::vector<float>> beta(n, std::vector<float>(m, 0.f));
    for (int i = 0; i < n; ++i) {
        float sum_L = 0.f;
        for (int j = 0; j < m; ++j) sum_L += L[i][j];
        float denom = cfg.clutter_density + cfg.p_detection * sum_L;
        for (int j = 0; j < m; ++j)
            if (L[i][j] > 0.f)
                beta[i][j] = cfg.p_detection * L[i][j] / denom;
    }

    // 4. Update each track with combined weighted innovation
    for (int i = 0; i < n; ++i) {
        float innov_x    = 0.f;
        float innov_y    = 0.f;
        float total_beta = 0.f;

        for (int j = 0; j < m; ++j) {
            if (beta[i][j] <= 0.f) continue;
            innov_x    += beta[i][j] * (detections[j].x - tracks[i].state[0]);
            innov_y    += beta[i][j] * (detections[j].y - tracks[i].state[1]);
            total_beta += beta[i][j];
        }

        kalman_update_combined(tracks[i], innov_x, innov_y, total_beta);
        tracks[i].last_update_s = timestamp_s;

        if (total_beta > 0.f) {
            ++tracks[i].frames_seen;
            tracks[i].frames_missed = 0;
        } else {
            ++tracks[i].frames_missed;
        }
    }

    // 5. Return detection indices with total association probability < 0.5
    std::vector<int> unassociated;
    for (int j = 0; j < m; ++j) {
        float total = 0.f;
        for (int i = 0; i < n; ++i) total += beta[i][j];
        if (total < 0.5f) unassociated.push_back(j);
    }
    return unassociated;
}
```

- [ ] **Step 4: Build and run tests**

```bash
cmake --build build --target heimdall_tracker_tests
cd build && ctest -R heimdall_tracker_tests --output-on-failure
```

Expected: all Kalman tests still pass, plus 7 new jpda tests — 14 passed total, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add src/tracker/jpda.cpp tests/test_jpda.cpp
git commit -m "feat: JPDAF association — gating, Gaussian likelihoods, combined innovations"
```

---

## Task 5: ObjectTracker Public Interface

**Files:**
- Modify: `src/tracker/tracker.cpp` (replace stub)
- Modify: `tests/test_tracker.cpp` (replace stub)

`ObjectTracker` wraps JPDA, manages track IDs, and handles the TENTATIVE → CONFIRMED → LOST lifecycle. This is the only class callers interact with.

- [ ] **Step 1: Write failing tests (replace tests/test_tracker.cpp stub)**

```cpp
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tracker/tracker.h"

static std::vector<FieldDetection> at(float x, float y, int cls = 0) {
    return {{cls, x, y, 0.9f}};
}

TEST_CASE("new detection produces no CONFIRMED event until confirmation_frames met", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 3});
    auto events = t.update(at(1.f, 2.f), 0.0);
    bool any_confirmed = std::any_of(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    REQUIRE_FALSE(any_confirmed);
}

TEST_CASE("track confirmed exactly at confirmation_frames", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 3});
    std::vector<TrackEvent> events;
    for (int i = 0; i < 3; ++i)
        events = t.update(at(1.f, 2.f), i * 0.02);
    long confirmed = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    REQUIRE(confirmed == 1);
}

TEST_CASE("confirmed track emits UPDATED every subsequent frame", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 2});
    for (int i = 0; i < 2; ++i)
        t.update(at(0.f, 0.f), i * 0.02);
    auto events = t.update(at(0.f, 0.f), 2 * 0.02);
    long updated = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::UPDATED; });
    REQUIRE(updated >= 1);
}

TEST_CASE("confirmed track emits LOST after loss_frames missed frames", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 2, .loss_frames = 3, .gate_distance = 0.1f});
    for (int i = 0; i < 2; ++i)
        t.update(at(0.f, 0.f), i * 0.02);
    std::vector<TrackEvent> events;
    for (int i = 0; i < 3; ++i)
        events = t.update({}, (2 + i) * 0.02);
    long lost = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::LOST; });
    REQUIRE(lost == 1);
}

TEST_CASE("after LOST, no further events for that track", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 1, .loss_frames = 1, .gate_distance = 0.1f});
    t.update(at(0.f, 0.f), 0.0);   // confirm immediately
    t.update({}, 0.02);              // miss — emit LOST
    auto events = t.update({}, 0.04);
    bool any = std::any_of(events.begin(), events.end(),
        [](const TrackEvent& e){
            return e.type == TrackEventType::LOST || e.type == TrackEventType::UPDATED;
        });
    REQUIRE_FALSE(any);
}

TEST_CASE("two well-separated objects tracked with correct class IDs", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 2, .gate_distance = 0.5f});
    std::vector<FieldDetection> two = {{0, 0.f, 0.f, 0.9f}, {1, 10.f, 10.f, 0.9f}};
    std::vector<TrackEvent> events;
    for (int i = 0; i < 2; ++i)
        events = t.update(two, i * 0.02);
    long confirmed = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    REQUIRE(confirmed == 2);
    std::vector<int> class_ids;
    for (const auto& e : events)
        if (e.type == TrackEventType::CONFIRMED)
            class_ids.push_back(e.object.class_id);
    std::sort(class_ids.begin(), class_ids.end());
    REQUIRE(class_ids == std::vector<int>{0, 1});
}

TEST_CASE("tentative track discarded on first missed frame", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 5, .gate_distance = 0.1f});
    t.update(at(0.f, 0.f), 0.0);   // creates tentative at (0,0)
    t.update({}, 0.02);              // miss — discard tentative
    // Confirm track at far location to check (0,0) isn't promoted later
    for (int i = 0; i < 5; ++i)
        t.update(at(0.f, 0.f), (2 + i) * 0.02);
    auto events = t.update(at(0.f, 0.f), 7 * 0.02);
    long confirmed = std::count_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    // Should be newly confirmed (from fresh tentative), not the discarded original
    REQUIRE(confirmed <= 1);
}

TEST_CASE("CONFIRMED event carries correct position", "[tracker]") {
    ObjectTracker t({.confirmation_frames = 2});
    std::vector<TrackEvent> events;
    for (int i = 0; i < 2; ++i)
        events = t.update(at(3.f, 7.f), i * 0.02);
    auto it = std::find_if(events.begin(), events.end(),
        [](const TrackEvent& e){ return e.type == TrackEventType::CONFIRMED; });
    REQUIRE(it != events.end());
    REQUIRE_THAT(it->object.x, Catch::Matchers::WithinAbs(3.f, 0.5f));
    REQUIRE_THAT(it->object.y, Catch::Matchers::WithinAbs(7.f, 0.5f));
}
```

- [ ] **Step 2: Build to confirm failure**

```bash
cmake --build build --target heimdall_tracker_tests 2>&1 | tail -5
```

Expected: link error — `ObjectTracker` not defined.

- [ ] **Step 3: Implement src/tracker/tracker.cpp (replace stub)**

```cpp
#include "tracker.h"
#include "kalman.h"
#include <algorithm>

ObjectTracker::ObjectTracker(Config config) : config_(config) {}

TrackedObject ObjectTracker::to_tracked_object(const Track& t) const {
    return {
        .track_id   = t.id,
        .class_id   = t.class_id,
        .x          = t.state[0],
        .y          = t.state[1],
        .vx         = t.state[2],
        .vy         = t.state[3],
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

    // Run JPDA — predicts, associates, updates all existing tracks
    auto unassociated = jpda_update(tracks_, detections, timestamp_s, jpda_cfg);

    // Create new tentative tracks for unassociated detections
    for (int idx : unassociated) {
        const auto& d = detections[idx];
        tracks_.push_back(make_track(next_id_++, d.class_id, d.x, d.y, d.confidence, timestamp_s));
    }

    std::vector<TrackEvent> events;
    std::vector<Track> surviving;

    for (auto& t : tracks_) {
        if (t.status == TrackStatus::TENTATIVE) {
            if (t.frames_missed > 0) {
                // Tentative tracks are discarded on first miss — no LOST event
                continue;
            }
            if (t.frames_seen >= config_.confirmation_frames) {
                t.status = TrackStatus::CONFIRMED;
                events.push_back({TrackEventType::CONFIRMED, to_tracked_object(t)});
            }
            // Still tentative — keep but don't emit UPDATED
            surviving.push_back(std::move(t));
            continue;
        }

        // CONFIRMED track
        if (t.frames_missed >= config_.loss_frames) {
            events.push_back({TrackEventType::LOST, to_tracked_object(t)});
            continue;  // drop from tracking
        }

        events.push_back({TrackEventType::UPDATED, to_tracked_object(t)});
        surviving.push_back(std::move(t));
    }

    tracks_ = std::move(surviving);
    return events;
}
```

- [ ] **Step 4: Build and run all tracker tests**

```bash
cmake --build build --target heimdall_tracker_tests
cd build && ctest -R heimdall_tracker_tests --output-on-failure
```

Expected: all 22 tests pass (7 kalman + 7 jpda + 8 tracker), 0 failed.

- [ ] **Step 5: Commit**

```bash
git add src/tracker/tracker.cpp tests/test_tracker.cpp
git commit -m "feat: ObjectTracker — JPDA-based track lifecycle with CONFIRMED/UPDATED/LOST events"
```

---

## Spike Complete — Criteria

| Check | Condition |
|-------|-----------|
| Build | `cmake --build build --target heimdall_tracker_tests` succeeds on any machine |
| Tests | `ctest -R heimdall_tracker_tests` — 22 passed, 0 failed |
| No hardware | All tasks build and test on Windows/Mac/Linux without Docker or Jetson |

**Milestone 4** wires the tracker to ZeroMQ + Protobuf: robot pose in via PUSH/PULL, tracked objects out. A `tools/mock_robot` program on the laptop simulates the RoboRIO interface for local testing.
