#define _USE_MATH_DEFINES
#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pose/pose_estimator.h"
#include "pose/camera_params.h"

using Catch::Matchers::WithinAbs;

// Overhead camera helper: R = diag(1,1,-1)
// cam_X->rob_X, cam_Y->rob_Y, cam_Z->-rob_Z (pointing straight down).
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
    std::array<float,9> R = rotation_from_euler(0.f, 0.f, 0.f);
    float rx = R[0*3+2], ry = R[1*3+2], rz = R[2*3+2];
    REQUIRE_THAT(rx, WithinAbs(1.f, 1e-5f));
    REQUIRE_THAT(ry, WithinAbs(0.f, 1e-5f));
    REQUIRE_THAT(rz, WithinAbs(0.f, 1e-5f));
}

TEST_CASE("rotation_from_euler pitch=pi/2 maps cam-Z to robot -Z (straight down)", "[euler]") {
    std::array<float,9> R = rotation_from_euler(0.f, static_cast<float>(M_PI / 2), 0.f);
    float rx = R[0*3+2], ry = R[1*3+2], rz = R[2*3+2];
    REQUIRE_THAT(rx, WithinAbs(0.f,  1e-5f));
    REQUIRE_THAT(ry, WithinAbs(0.f,  1e-5f));
    REQUIRE_THAT(rz, WithinAbs(-1.f, 1e-5f));
}

TEST_CASE("rotation_from_euler yaw=pi/2 maps cam-Z to robot +Y (left side)", "[euler]") {
    std::array<float,9> R = rotation_from_euler(static_cast<float>(M_PI / 2), 0.f, 0.f);
    float rx = R[0*3+2], ry = R[1*3+2], rz = R[2*3+2];
    REQUIRE_THAT(rx, WithinAbs(0.f, 1e-5f));
    REQUIRE_THAT(ry, WithinAbs(1.f, 1e-5f));
    REQUIRE_THAT(rz, WithinAbs(0.f, 1e-5f));
}

// ── PoseEstimator ──────────────────────────────────────────────────────────

TEST_CASE("principal-point pixel projects to camera ground footprint", "[pose]") {
    // Overhead camera at (0,0,2m), robot at (5,3), heading=0.
    // px=cx=320, py=cy=240 -> u=0, v=0 -> d_rob=[0,0,-1] -> ground at (5,3)
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    Detection det = make_det(0, 319.f, 240.f, 2.f, 0.f); // px=320=cx, py=240=cy
    RobotPose pose{5.f, 3.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    REQUIRE_THAT(results[0].x, WithinAbs(5.f, 0.01f));
    REQUIRE_THAT(results[0].y, WithinAbs(3.f, 0.01f));
}

TEST_CASE("pixel offset maps to correct field offset (heading=0)", "[pose]") {
    // Camera overhead (0,0,2m). px=370, py=240 -> u=0.1, v=0
    // d_rob=[0.1,0,-1], d_field=[0.1,0,-1], t=2 -> x=0.2, y=0
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    Detection det = make_det(0, 369.f, 240.f, 2.f, 0.f); // px=370=cx+50
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    REQUIRE_THAT(results[0].x, WithinAbs(0.2f, 0.01f));
    REQUIRE_THAT(results[0].y, WithinAbs(0.f,  0.01f));
}

TEST_CASE("camera extrinsic offset shifts ground projection", "[pose]") {
    // Camera at (1m forward, 0, 2m) in robot frame, robot at origin.
    // Principal point -> x=1, y=0 (directly below camera)
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
    // Camera at (1,0,2) in robot frame. Robot at (5,3), heading=pi/2.
    // Camera 1m forward -> 1m in field +Y (robot forward = field +Y at heading=pi/2).
    // Ground point: (5, 4)
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
    // R = identity: cam_Z (forward) = rob_Z (up) -> ray points up -> no ground intersection
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
    // Camera overhead (0,0,2m). Detection: top=100, height=200.
    // bottom-center: py = 100+200 = 300. With cy=240, fy=500: v=(300-240)/500=0.12
    // y = 2 * 0.12 = 0.24
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    Detection det = make_det(0, 319.f, 100.f, 2.f, 200.f);
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    REQUIRE_THAT(results[0].y, WithinAbs(0.24f, 0.01f));
}

TEST_CASE("class_id and confidence are preserved", "[pose]") {
    auto cam = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, 0.f, 2.f);
    PoseEstimator estimator({cam});

    Detection det{0, 7, 0.83f, 319.f, 240.f, 2.f, 0.f, 0};
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project({det}, pose);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].class_id == 7);
    REQUIRE_THAT(results[0].confidence, WithinAbs(0.83f, 1e-5f));
}

// ── rotation / flip handling ─────────────────────────────────────────────────

// A forward-facing, pitched-down camera like cam1: 180° pipeline rotation, real intrinsics.
static CameraParams cam1_like(int rotation) {
    CameraParams p;
    p.intrinsics = {539.8783f, 541.1769f, 353.8330f, 195.5427f,
                    -0.001437f, 0.081865f, -0.004672f, -0.002202f, -0.158375f};
    p.extrinsics.tx = 0.0f; p.extrinsics.ty = 0.0889f; p.extrinsics.tz = 0.5207f;
    p.extrinsics.R  = rotation_from_euler(0.4363f, 0.5061f, 0.0f);
    p.rotation = rotation; p.width = 640; p.height = 480;
    return p;
}

TEST_CASE("rotation=180 un-rotates bbox to the native-frame ground contact", "[pose][rotation]") {
    // A native-frame bbox (L,T,w,h). Under a 180° rotation the same object's bbox in the
    // nvinfer frame is (W-1-L-w, H-1-T-h, w, h). project() with rotation=180 must recover the
    // exact same field point that rotation=0 gets from the native bbox — i.e. the un-rotation
    // is the exact inverse. This is the regression guard for the flip-intrinsic sign bugs.
    const float L = 279.f, T = 169.f, w = 60.f, h = 60.f;   // native bbox
    Detection native_det = make_det(0, L, T, w, h);
    Detection rot_det     = make_det(0, 640.f - 1.f - (L + w), 480.f - 1.f - (T + h), w, h);
    RobotPose pose{0.f, 0.f, 0.f};

    auto r0   = PoseEstimator({cam1_like(0)}).project({native_det}, pose);
    auto r180 = PoseEstimator({cam1_like(180)}).project({rot_det},  pose);
    REQUIRE(r0.size() == 1);
    REQUIRE(r180.size() == 1);
    REQUIRE_THAT(r180[0].x, WithinAbs(r0[0].x, 1e-3f));
    REQUIRE_THAT(r180[0].y, WithinAbs(r0[0].y, 1e-3f));
}

TEST_CASE("closer object projects nearer under 180° rotation", "[pose][rotation]") {
    // As the robot approaches a stationary floor object its bbox grows and moves down in the
    // rotated frame; the reported forward distance must DECREASE. Before the fix this diverged
    // (object's head was used as ground contact), which a real drive-test caught (#27).
    PoseEstimator est({cam1_like(180)});
    RobotPose pose{0.f, 0.f, 0.f};
    auto far  = est.project({make_det(0, 300.f, 250.f, 60.f, 60.f)}, pose);
    auto near = est.project({make_det(0, 295.f, 210.f, 79.f, 79.f)}, pose);
    REQUIRE(far.size() == 1);
    REQUIRE(near.size() == 1);
    REQUIRE(far[0].x  > 0.f);
    REQUIRE(near[0].x > 0.f);
    REQUIRE(near[0].x < far[0].x);   // closer bbox -> nearer forward distance
}

TEST_CASE("multiple cameras projected independently", "[pose]") {
    auto cam0 = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f,  0.f, 2.f);
    auto cam1 = overhead_cam(500.f, 500.f, 320.f, 240.f, 0.f, -1.f, 2.f); // 1m in -Y (right)
    PoseEstimator estimator({cam0, cam1});

    std::vector<Detection> dets = {
        {0, 0, 0.9f, 319.f, 240.f, 2.f, 0.f, 0},
        {1, 1, 0.8f, 319.f, 240.f, 2.f, 0.f, 0},
    };
    RobotPose pose{0.f, 0.f, 0.f};

    auto results = estimator.project(dets, pose);
    REQUIRE(results.size() == 2);
    REQUIRE_THAT(results[0].x, WithinAbs(0.f, 0.01f));
    REQUIRE_THAT(results[0].y, WithinAbs(0.f, 0.01f));
    REQUIRE_THAT(results[1].x, WithinAbs(0.f,  0.01f));
    REQUIRE_THAT(results[1].y, WithinAbs(-1.f, 0.01f));
}
