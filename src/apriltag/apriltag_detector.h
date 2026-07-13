#pragma once
#include "tag_layout.h"
#include <cstdint>
#include <optional>

struct VisionPoseResult {
    double   x, y;             // field-relative robot position, meters
    double   heading_rad;      // field-relative heading, radians CCW from +X
    uint64_t timestamp_ns;     // capture time (CLOCK_MONOTONIC)

    // Quality metadata for the robot's dynamic std devs (§2B).
    uint32_t tag_count        = 1;    // tags used in this solve
    double   avg_tag_distance = 0.0;  // mean tag distance, meters
    double   reproj_error     = 0.0;  // mean reprojection error, px
    double   ambiguity        = 0.0;  // IPPE ambiguity ratio (0 for constrained solve)
    uint32_t solve_mode       = 0;    // 0 = gyro-constrained, 1 = IPPE fallback
};

class AprilTagDetector {
public:
    explicit AprilTagDetector(AprilTagLayout layout);
    ~AprilTagDetector();

    // Feed a robot pose sample into the gyro history buffer (call at ~50 Hz from pose_recv_loop).
    // yaw: field-relative heading radians; vyaw: rad/s CCW positive; timestamp_ns: Jetson CLOCK_MONOTONIC.
    void update_gyro(double yaw, double vyaw, uint64_t timestamp_ns);

    // Capture one frame, detect tags, solve pose.
    //
    // The odometry/gyro yaw fed via update_gyro() is not trusted as an absolute
    // field heading until self-calibrated: until then, every solve uses the
    // unconstrained IPPE_SQUARE solver (+ floor-constraint disambiguation). Once
    // a run of consistent low-ambiguity IPPE solves agrees on an odometry->field
    // yaw offset, the fast gyro-constrained linear solve takes over, corrected
    // by that offset — so accuracy no longer depends on a precise robot lineup
    // or manual heading zero before the match. Set force_unconstrained_solver in
    // AprilTagLayout to always use IPPE and skip the constrained solve entirely.
    std::optional<VisionPoseResult> detect();

    bool is_open() const;

private:
    struct Impl;
    Impl* impl_;
};
