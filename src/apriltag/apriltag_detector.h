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
    // Uses constrained linear solve (rotation from gyro) when gyro stream is fresh;
    // falls back to IPPE_SQUARE + floor-constraint disambiguation otherwise.
    std::optional<VisionPoseResult> detect();

    bool is_open() const;

private:
    struct Impl;
    Impl* impl_;
};
