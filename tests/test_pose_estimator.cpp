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
