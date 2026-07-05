#pragma once
#include <string>
#include <unordered_map>
#include <array>

// Field-space pose of a single AprilTag
struct TagPose {
    double x, y, z;          // meters 
    double roll, pitch, yaw; // radians
};

// Camera intrinsics for the AprilTag camera.
struct AprilTagCameraParams {
    std::string device;
    int width  = 640;
    int height = 480;
    int fps    = 10;
    double fx = 600, fy = 600, cx = 320, cy = 240; // intrinsics
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0; // distortion
};

// Fixed transform from robot origin to AprilTag camera (meters + radians).
struct RobotToCameraTransform {
    double x, y, z;          // meters 
    double roll, pitch, yaw; // radians
};

struct AprilTagLayout {
    double                             tag_size_meters = 0.1651;
    AprilTagCameraParams               camera;
    RobotToCameraTransform             robot_to_camera;
    std::unordered_map<int, TagPose>   tags;

    // When true, always use the unconstrained IPPE_SQUARE solver — never the
    // gyro/odometry-constrained solve. Use this if the robot's heading feed
    // isn't trustworthy enough to seed the constrained solve (e.g. no yaw
    // calibration source, or debugging pose accuracy independent of odometry).
    bool                               force_unconstrained_solver = false;
};

// Throws std::runtime_error if the file is missing or malformed.
AprilTagLayout load_apriltag_layout(const std::string& path);
