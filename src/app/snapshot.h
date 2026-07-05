#pragma once
#include "pipeline/detection.h"
#include "pose/camera_params.h"
#include "tracker/field_detection.h"
#include "tracker/track_event.h"
#include <cstdint>
#include <string>
#include <vector>

// Debug snapshot: dump one pipeline frame's full state — raw inputs, projected
// intermediate, and final outputs — to a single JSON file. Used to correlate a
// bad output with the exact inputs that produced it.
namespace snapshot {

struct Frame {
    uint64_t frame_ts_ns          = 0;  // Detection::timestamp_ns (pipeline running time)
    uint64_t capture_monotonic_ns = 0;  // absolute capture time
    RobotPose                    robot_pose{};       // AprilTag-derived pose used for projection
    std::vector<Detection>       detections;         // raw fuel pixel coords (input)
    std::vector<FieldDetection>  field_detections;   // projected field coords (intermediate)
    std::vector<TrackEvent>      events;             // tracker output (end of pipeline)
    bool healthy = true;
};

// Writes `frame` to <dir>/snapshot_<frame_ts_ns>.json, creating `dir` if needed.
// Returns the written path, or "" on failure.
std::string write(const std::string& dir, const Frame& frame);

}  // namespace snapshot
