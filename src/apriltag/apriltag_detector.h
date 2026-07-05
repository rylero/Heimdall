#pragma once
#include "tag_layout.h"
#include <cstdint>
#include <optional>

struct VisionPoseResult {
    double   x, y;         // field-relative robot position, meters
    double   heading_rad;  // field-relative heading, radians CCW from +X
    uint64_t timestamp_ns; // capture time (CLOCK_MONOTONIC)
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
