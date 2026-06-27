#pragma once
#include "tag_layout.h"
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

    // Captures one frame, detects tags, runs PnP.
    // Returns the best (lowest-ambiguity) pose estimate, or nullopt if nothing reliable.
    std::optional<VisionPoseResult> detect(double current_robot_x, double current_robot_y);

    bool is_open() const;

private:
    struct Impl;
    Impl* impl_;
};
