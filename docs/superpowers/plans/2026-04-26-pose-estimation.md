# Heimdall Milestone 2: Ground Ray Pose Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pure C++20 `PoseEstimator` class that projects camera-space `Detection` bounding boxes onto the field ground plane, outputting field-relative `FieldDetection` positions for the JPDA tracker.

**Architecture:** Ground ray method — for each detection, unproject the bounding-box bottom-center pixel through the camera intrinsic matrix to get a ray in camera space, transform via camera extrinsics to robot frame, rotate by robot heading to field frame, then intersect with the z=0 ground plane. A `heimdall_pose` CMake library target has no GStreamer/DeepStream dependency and is fully unit-testable on any machine.

**Tech Stack:** C++20, CMake (no additional dependencies), Catch2 v3 (already in project via FetchContent)

---

## Coordinate Systems

| Frame | X | Y | Z |
|-------|---|---|---|
| Camera (OpenCV) | right | down | forward (into scene) |
| Robot (WPiLib) | forward | left | up |
| Field (WPiLib) | toward opponent wall | left | up |

**Camera → Robot** transform: stored as a 3×3 rotation matrix `R[9]` (row-major) plus translation `(tx, ty, tz)`.

For a level, forward-facing camera, the base rotation is:
```
R_base = [[0,  0, 1],   # cam_Z (forward) → robot_X (forward)
           [-1, 0, 0],   # cam_X (right)   → -robot_Y (right = -left)
           [0, -1, 0]]   # cam_Y (down)    → -robot_Z (down = -up)
```

**Ground ray algorithm:**
1. Bottom-center pixel: `px = left + width/2`, `py = top + height`
2. Normalised camera direction: `d_cam = [(px-cx)/fx, (py-cy)/fy, 1]`
3. Robot frame: `d_rob = R * d_cam`, camera origin `o_rob = (tx, ty, tz)`
4. Field frame: `d_field = R_heading * d_rob`, `o_field = (robot_x + R_heading * o_rob.xy, tz)`
5. Ground intersection (z=0): `t = -tz / d_field.z` (skip if `d_field.z >= 0`)
6. Result: `(o_field.x + t * d_field.x, o_field.y + t * d_field.y)`

---

## File Structure

```
src/pose/
  camera_params.h       # CameraIntrinsics, CameraExtrinsics, CameraParams, RobotPose
                        # + rotation_from_euler() helper
  pose_estimator.h      # PoseEstimator class declaration
  pose_estimator.cpp    # Ground ray implementation
tests/
  test_pose_estimator.cpp
CMakeLists.txt          # Modify: add heimdall_pose library (no hardware dep)
tests/CMakeLists.txt    # Modify: add heimdall_pose_tests executable
```

---

## Task 1: CMake Scaffold

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Create: `src/pose/pose_estimator.cpp` (stub)
- Create: `tests/test_pose_estimator.cpp` (stub)

- [ ] **Step 1: Add heimdall_pose to CMakeLists.txt**

In `CMakeLists.txt`, add this block immediately after the `heimdall_tracker` block and before the GStreamer section:

```cmake
# ---------------------------------------------------------------------------
# Pose estimation library — no hardware dependency, unit-testable anywhere
# ---------------------------------------------------------------------------
add_library(heimdall_pose
    src/pose/pose_estimator.cpp
)
target_include_directories(heimdall_pose PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
```

- [ ] **Step 2: Update tests/CMakeLists.txt**

Add the pose test executable after the existing `heimdall_tracker_tests` block:

```cmake
add_executable(heimdall_pose_tests
    test_pose_estimator.cpp
)
target_link_libraries(heimdall_pose_tests
    heimdall_pose
    Catch2::Catch2WithMain
)
catch_discover_tests(heimdall_pose_tests)
```

- [ ] **Step 3: Create stub files**

`src/pose/pose_estimator.cpp`:
```cpp
// Stub — implemented in Task 3.
```

`tests/test_pose_estimator.cpp`:
```cpp
// Stub — implemented in Task 3.
```

- [ ] **Step 4: Verify CMake configures**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target help 2>&1 | grep -i pose
```

Expected: `heimdall_pose` and `heimdall_pose_tests` both appear.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt tests/CMakeLists.txt \
        src/pose/pose_estimator.cpp \
        tests/test_pose_estimator.cpp
git commit -m "feat: add heimdall_pose CMake target and pose test executable"
```

---

## Task 2: Camera Params Types

**Files:**
- Create: `src/pose/camera_params.h`

Pure header. No test file (types tested through PoseEstimator in Task 3). The `rotation_from_euler` helper IS tested in Task 3.

- [ ] **Step 1: Create src/pose/camera_params.h**

```cpp
#pragma once
#include <array>
#include <cmath>

struct CameraIntrinsics {
    float fx, fy;   // focal lengths, pixels
    float cx, cy;   // principal point, pixels
};

struct CameraExtrinsics {
    float tx, ty, tz;       // camera position in robot frame, meters (tz = height above ground)
    std::array<float, 9> R; // rotation matrix camera→robot, row-major 3x3
};

struct CameraParams {
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
};

// Robot position and heading in field frame (WPiLib standard coordinates).
struct RobotPose {
    float x, y;       // field-relative, meters
    float heading;    // radians, CCW positive from +X axis
};

// Build a camera→robot rotation matrix from mounting angles.
//
// Convention (camera frame = OpenCV: X right, Y down, Z forward):
//   yaw   — which direction the camera faces on the robot, radians CCW from robot +X.
//            yaw=0 → forward, yaw=pi/2 → left side, yaw=-pi/2 → right side.
//   pitch — downward tilt of optical axis, radians. pitch=0 → level, pitch>0 → looking down.
//   roll  — rotation around optical axis, radians (usually 0).
//
// Returns 9-element row-major 3x3 rotation matrix.
inline std::array<float, 9> rotation_from_euler(float yaw, float pitch, float roll) {
    // Base rotation: level, forward-facing camera (yaw=0, pitch=0, roll=0)
    //   cam_Z (forward) → robot_X,  cam_X (right) → -robot_Y,  cam_Y (down) → -robot_Z
    const float R_base[9] = {
         0.f,  0.f, 1.f,
        -1.f,  0.f, 0.f,
         0.f, -1.f, 0.f,
    };

    // Rx(-pitch): pitch down rotates around cam_X toward cam_Y (down)
    const float cp = std::cos(-pitch), sp = std::sin(-pitch);
    const float Rx[9] = {
        1.f,  0.f,  0.f,
        0.f,  cp,  -sp,
        0.f,  sp,   cp,
    };

    // Rz(roll): roll around cam_Z
    const float cr = std::cos(roll), sr = std::sin(roll);
    const float Rz_cam[9] = {
         cr, -sr, 0.f,
         sr,  cr, 0.f,
        0.f, 0.f, 1.f,
    };

    // Rz(yaw): mount orientation around robot_Z
    const float cy2 = std::cos(yaw), sy2 = std::sin(yaw);
    const float Rz_rob[9] = {
         cy2, -sy2, 0.f,
         sy2,  cy2, 0.f,
        0.f,   0.f, 1.f,
    };

    // Helper: C = A * B (3x3 row-major)
    auto mat3mul = [](const float A[9], const float B[9], float C[9]) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                C[i*3+j] = 0.f;
                for (int k = 0; k < 3; ++k)
                    C[i*3+j] += A[i*3+k] * B[k*3+j];
            }
    };

    // R = Rz_rob * R_base * Rx(-pitch) * Rz_cam(roll)
    float tmp1[9], tmp2[9], result[9];
    mat3mul(R_base, Rx, tmp1);
    mat3mul(tmp1, Rz_cam, tmp2);
    mat3mul(Rz_rob, tmp2, result);

    std::array<float, 9> out;
    for (int i = 0; i < 9; ++i) out[i] = result[i];
    return out;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/pose/camera_params.h
git commit -m "feat: camera params types and rotation_from_euler helper"
```

---

## Task 3: PoseEstimator + Tests

**Files:**
- Modify: `src/pose/pose_estimator.cpp` (replace stub)
- Create: `src/pose/pose_estimator.h`
- Modify: `tests/test_pose_estimator.cpp` (replace stub)

- [ ] **Step 1: Write failing tests**

Replace `tests/test_pose_estimator.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pose/pose_estimator.h"
#include "pose/camera_params.h"

using Catch::Matchers::WithinAbs;

// Straight-down camera helper: R = diag(1, 1, -1)
// cam_X→rob_X, cam_Y→rob_Y, cam_Z→-rob_Z (pointing straight down).
// Used only for tests — not a physically realistic mounting.
static CameraParams overhead_cam(float fx, float fy, float cx, float cy,
                                  float tx, float ty, float tz) {
    return {
        .intrinsics = {fx, fy, cx, cy},
        .extrinsics = {tx, ty, tz, {1.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f,-1.f}},
    };
}

static Detection make_det(int cam_id, float left, float top, float w, float h) {
    return {cam_id, 0, 0.9f, left, top, w, h, 0};
}

// ── rotation_from_euler ────────────────────────────────────────────────────

TEST_CASE("rotation_from_euler yaw=0 pitch=0 maps cam-Z to robot-X", "[euler]") {
    auto R = rotation_from_euler(0.f, 0.f, 0.f);
    // R * [0,0,1] = robot forward (+X)
    float dx = R[0*3+2], dy = R[1*3+2], dz = R[2*3+2];
    REQUIRE_THAT(dx, WithinAbs(1.f, 1e-5f));
    REQUIRE_THAT(dy, WithinAbs(0.f, 1e-5f));
    REQUIRE_THAT(dz, WithinAbs(0.f, 1e-5f));
}

TEST_CASE("rotation_from_euler pitch=pi/2 maps cam-Z to robot -Z (straight down)", "[euler]") {
    auto R = rotation_from_euler(0.f, static_cast<float>(M_PI / 2), 0.f);
    float dx = R[0*3+2], dy = R[1*3+2], dz = R[2*3+2];
    REQUIRE_THAT(dx, WithinAbs(0.f, 1e-5f));
    REQUIRE_THAT(dy, WithinAbs(0.f, 1e-5f));
    REQUIRE_THAT(dz, WithinAbs(-1.f, 1e-5f));
}

TEST_CASE("rotation_from_euler yaw=pi/2 maps cam-Z to robot +Y (left side)", "[euler]") {
    auto R = rotation_from_euler(static_cast<float>(M_PI / 2), 0.f, 0.f);
    float dx = R[0*3+2], dy = R[1*3+2], dz = R[2*3+2];
    REQUIRE_THAT(dx, WithinAbs(0.f, 1e-5f));
    REQUIRE_THAT(dy, WithinAbs(1.f, 1e-5f));
    REQUIRE_THAT(dz, WithinAbs(0.f, 1e-5f));
}

// ── PoseEstimator ──────────────────────────────────────────────────────────

TEST_CASE("principal-point pixel projects to camera's ground footprint", "[pose]") {
    // Overhead camera at (0,0,2m), robot at (5,3), heading=0.
    // Principal point (cx,cy) bottom-center (py=cy with height=0 for simplicity):
    // detection: left=319, top=240, w=2, h=0 → px=320=cx, py=240=cy
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    Detection det = make_det(0, 319.f, 240.f, 2.f, 0.f); // px=320=cx, py=240=cy
    RobotPose pose{5.f, 3.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    // Camera is directly above robot center, so ground point ≈ robot position
    REQUIRE_THAT(results[0].x, WithinAbs(5.f, 0.01f));
    REQUIRE_THAT(results[0].y, WithinAbs(3.f, 0.01f));
}

TEST_CASE("pixel offset maps to correct field offset (heading=0)", "[pose]") {
    // Camera overhead at (0,0,2m), 100px right of center → 0.4m in robot +X direction.
    // At height 2m, 1px = 2/fx meters = 2/500 = 0.004m; 100px = 0.4m
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    // left=369, top=239, w=2, h=1 → px=370=cx+50, py=240=cy
    // u=(370-320)/500=0.1, v=0 → d_rob=[0.1,0,-1] → d_field=[0.1,0,-1]
    // t=2, x=0+2*0.1=0.2, y=0
    Detection det = make_det(0, 369.f, 240.f, 2.f, 0.f);
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    REQUIRE_THAT(results[0].x, WithinAbs(0.2f, 0.01f));
    REQUIRE_THAT(results[0].y, WithinAbs(0.f,  0.01f));
}

TEST_CASE("camera extrinsic offset shifts ground projection", "[pose]") {
    // Camera at (1m forward, 0, 2m height) in robot frame, robot at origin.
    // Principal point projects to (1, 0) — directly below camera.
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 1.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    Detection det = make_det(0, 319.f, 240.f, 2.f, 0.f);
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    REQUIRE_THAT(results[0].x, WithinAbs(1.f, 0.01f));
    REQUIRE_THAT(results[0].y, WithinAbs(0.f, 0.01f));
}

TEST_CASE("robot heading rotates field projection", "[pose]") {
    // Camera at (1,0,2) in robot frame. Robot at (5,3), heading=pi/2 (facing field +Y).
    // Camera 1m forward from robot → 1m in field +Y direction (robot forward = field +Y when heading=pi/2).
    // Principal-point ground: below camera → field (5, 3+1) = (5, 4).
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 1.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    Detection det = make_det(0, 319.f, 240.f, 2.f, 0.f);
    RobotPose pose{5.f, 3.f, static_cast<float>(M_PI / 2)};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    REQUIRE_THAT(results[0].x, WithinAbs(5.f, 0.01f));
    REQUIRE_THAT(results[0].y, WithinAbs(4.f, 0.01f));
}

TEST_CASE("upward-pointing ray produces no detection", "[pose]") {
    // R = identity: cam_Z (forward) = rob_Z (up). Ray points up → no ground intersection.
    CameraParams cam{
        .intrinsics = {500.f, 500.f, 320.f, 240.f},
        .extrinsics = {0.f, 0.f, 1.f, {1,0,0, 0,1,0, 0,0,1}},
    };
    PoseEstimator estimator({cam});

    Detection det = make_det(0, 319.f, 240.f, 2.f, 0.f);
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.empty());
}

TEST_CASE("bottom-center pixel used (not bbox center)", "[pose]") {
    // Camera overhead (0,0,2m). Detection with height=100px.
    // Center would be py = top+h/2 = 200, bottom-center py = top+h = 300.
    // With cy=240, fy=500:
    //   center:        v = (200-240)/500 = -0.08  → y = -0.16m
    //   bottom-center: v = (300-240)/500 = +0.12  → y = +0.24m
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    // left=319, top=100, w=2, h=200 → bottom-center: px=320, py=300
    Detection det = make_det(0, 319.f, 100.f, 2.f, 200.f);
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    // py=300, cy=240, fy=500 → v=0.12 → y = 2 * 0.12 = 0.24
    REQUIRE_THAT(results[0].y, WithinAbs(0.24f, 0.01f));
}

TEST_CASE("class_id and confidence are preserved", "[pose]") {
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    Detection det{0, 7, 0.83f, 319.f, 240.f, 2.f, 0.f, 0};
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].class_id  == 7);
    REQUIRE_THAT(results[0].confidence, WithinAbs(0.83f, 1e-5f));
}

TEST_CASE("multiple detections from multiple cameras projected independently", "[pose]") {
    auto cam0 = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f,  0.f, 2.f);
    auto cam1 = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, -1.f, 2.f); // 1m right
    PoseEstimator estimator({cam0, cam1});

    // Both detections at principal point → project to directly below each camera
    std::vector<Detection> dets = {
        {0, 0, 0.9f, 319.f, 240.f, 2.f, 0.f, 0},
        {1, 1, 0.8f, 319.f, 240.f, 2.f, 0.f, 0},
    };
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project(dets, pose);
    REQUIRE(results.size() == 2);
    REQUIRE_THAT(results[0].x, WithinAbs(0.f, 0.01f));
    REQUIRE_THAT(results[0].y, WithinAbs(0.f, 0.01f));
    REQUIRE_THAT(results[1].x, WithinAbs(0.f, 0.01f));
    REQUIRE_THAT(results[1].y, WithinAbs(-1.f, 0.01f)); // cam1 is 1m right (-rob_Y in WPiLib)
}
```

- [ ] **Step 2: Build to confirm failure**

```bash
cmake --build build --target heimdall_pose_tests 2>&1 | tail -5
```

Expected: compile error — `pose_estimator.h` not found.

- [ ] **Step 3: Create src/pose/pose_estimator.h**

```cpp
#pragma once
#include "camera_params.h"
#include "pipeline/detection.h"
#include "tracker/field_detection.h"
#include <vector>

class PoseEstimator {
public:
    explicit PoseEstimator(std::vector<CameraParams> cameras);

    // Project camera-space detections to field-relative positions.
    // Detections whose ray does not intersect the ground (pointing up) are silently dropped.
    std::vector<FieldDetection> project(
        const std::vector<Detection>& detections,
        const RobotPose&              robot_pose
    ) const;

private:
    std::vector<CameraParams> cameras_;

    // Returns false if the ray does not intersect the ground plane (z >= 0 direction).
    bool project_pixel(int camera_id,
                       float px, float py,
                       const RobotPose& robot_pose,
                       float& field_x, float& field_y) const;
};
```

- [ ] **Step 4: Implement src/pose/pose_estimator.cpp**

```cpp
#include "pose_estimator.h"
#include <cmath>

PoseEstimator::PoseEstimator(std::vector<CameraParams> cameras)
    : cameras_(std::move(cameras)) {}

bool PoseEstimator::project_pixel(int camera_id,
                                   float px, float py,
                                   const RobotPose& rp,
                                   float& field_x, float& field_y) const {
    const auto& cam  = cameras_[camera_id];
    const auto& intr = cam.intrinsics;
    const auto& extr = cam.extrinsics;
    const auto& R    = extr.R;  // camera→robot, row-major 3x3

    // 1. Unproject pixel to direction in camera frame
    const float u = (px - intr.cx) / intr.fx;
    const float v = (py - intr.cy) / intr.fy;
    // d_cam = [u, v, 1]

    // 2. Rotate direction to robot frame: d_rob = R * d_cam
    const float drx = R[0]*u + R[1]*v + R[2];
    const float dry = R[3]*u + R[4]*v + R[5];
    const float drz = R[6]*u + R[7]*v + R[8];

    // 3. Rotate direction to field frame via robot heading
    const float ch = std::cos(rp.heading), sh = std::sin(rp.heading);
    const float dfx = ch * drx - sh * dry;
    const float dfy = sh * drx + ch * dry;
    const float dfz = drz;

    // Ray must point downward to intersect ground
    if (dfz >= 0.f) return false;

    // 4. Camera origin in field frame
    const float ox_rob = extr.tx, oy_rob = extr.ty;
    const float ofx = rp.x + ch * ox_rob - sh * oy_rob;
    const float ofy = rp.y + sh * ox_rob + ch * oy_rob;
    const float ofz = extr.tz;  // camera height above ground

    // 5. Ground intersection: ofz + t * dfz = 0
    const float t = -ofz / dfz;
    field_x = ofx + t * dfx;
    field_y = ofy + t * dfy;
    return true;
}

std::vector<FieldDetection> PoseEstimator::project(
    const std::vector<Detection>& detections,
    const RobotPose&              robot_pose
) const {
    std::vector<FieldDetection> results;
    results.reserve(detections.size());

    for (const auto& det : detections) {
        if (det.camera_id < 0 || det.camera_id >= static_cast<int>(cameras_.size()))
            continue;

        // Use bottom-center of bounding box as ground contact point
        const float px = det.left + det.width  / 2.f;
        const float py = det.top  + det.height;

        float fx, fy;
        if (!project_pixel(det.camera_id, px, py, robot_pose, fx, fy))
            continue;

        results.push_back({det.class_id, fx, fy, det.confidence});
    }

    return results;
}
```

- [ ] **Step 5: Build and run all pose tests**

```bash
cmake --build build --target heimdall_pose_tests
./build/tests/Debug/heimdall_pose_tests.exe
```

Expected:
```
All tests passed (N assertions in 11 test cases)
```

Fix any failures before committing.

- [ ] **Step 6: Commit**

```bash
git add src/pose/pose_estimator.h src/pose/pose_estimator.cpp tests/test_pose_estimator.cpp
git commit -m "feat: ground ray pose estimator with multi-camera support"
```

---

## Milestone Complete — Criteria

| Check | Condition |
|-------|-----------|
| Build | `cmake --build build --target heimdall_pose_tests` succeeds on any machine |
| Tests | 11 tests pass (3 euler + 8 pose) |
| No hardware | All tasks build and test without Docker or Jetson |

**Milestone 4** wires `PoseEstimator` + `ObjectTracker` into the ZeroMQ communication layer: robot pose in → project detections → track → publish.

---

## Self-Review Notes

**Spec coverage:**
- ✅ Ground ray method
- ✅ Fixed camera mounts (extrinsics stored as R + translation)
- ✅ Field-relative WPiLib output
- ✅ Manual extrinsic calibration (params set by caller — web UI in M5)
- ✅ rotation_from_euler helper for web UI integration
- ✅ Multi-camera support (camera_id dispatch)
- ✅ Bottom-center pixel (bbox ground contact point)
- ✅ Outputs FieldDetection for JPDA tracker

**Coordinate convention documented inline in camera_params.h and this plan.**
