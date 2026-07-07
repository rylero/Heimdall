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

// AprilTag detector + pose-solve tuning. Previously hardcoded in apriltag_detector.cpp;
// exposed here so they can be tuned per venue without a recompile.
struct AprilTagSolverParams {
    // apriltag_detector tuning
    float  quad_decimate = 2.0f;   // input downsample: higher = faster but shorter detection range
    float  quad_sigma    = 0.0f;   // gaussian blur (sigma) before detection; >0 helps noisy images
    int    nthreads      = 2;      // detector worker threads
    bool   refine_edges  = true;   // sub-pixel corner refinement

    // Pose disambiguation (IPPE fallback)
    double ambiguity_max            = 0.25;  // reject IPPE solves with error-ratio above this
    double floor_disambiguation_min = 0.15;  // above this ratio, pick the on-floor (z≈0) solution
};

struct AprilTagLayout {
    double                             tag_size_meters = 0.1651;
    AprilTagCameraParams               camera;
    RobotToCameraTransform             robot_to_camera;
    AprilTagSolverParams               solver;
    std::unordered_map<int, TagPose>   tags;

    // When true, always use the unconstrained IPPE_SQUARE solver — never the
    // gyro/odometry-constrained solve. Use this if the robot's heading feed
    // isn't trustworthy enough to seed the constrained solve (e.g. no yaw
    // calibration source, or debugging pose accuracy independent of odometry).
    bool                               force_unconstrained_solver = false;
};

// Throws std::runtime_error if the file is missing or malformed.
AprilTagLayout load_apriltag_layout(const std::string& path);
