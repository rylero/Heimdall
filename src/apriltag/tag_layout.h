#pragma once
#include <string>
#include <unordered_map>
#include <array>

// Field-space pose of a single AprilTag (tag center, meters + radians).
struct TagPose {
    double x, y, z;
    double roll, pitch, yaw; // radians
};

// Camera intrinsics for the AprilTag camera.
struct AprilTagCameraParams {
    std::string device;
    int width  = 640;
    int height = 480;
    int fps    = 10;
    double fx = 600, fy = 600, cx = 320, cy = 240; // intrinsics (pixels)
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0; // distortion
};

// Fixed transform from robot origin to AprilTag camera (meters + radians).
struct RobotToCameraTransform {
    double x, y, z;
    double roll, pitch, yaw;
};

struct AprilTagLayout {
    double                             tag_size_meters = 0.1651;
    AprilTagCameraParams               camera;
    RobotToCameraTransform             robot_to_camera;
    std::unordered_map<int, TagPose>   tags; // tag_id → field pose
};

// Throws std::runtime_error if the file is missing or malformed.
AprilTagLayout load_apriltag_layout(const std::string& path);
