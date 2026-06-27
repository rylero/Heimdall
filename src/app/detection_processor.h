#pragma once
#include "app/detection_output.h"
#include "app/pose_buffer.h"
#include "comm/comm_layer.h"
#include "pipeline/detection.h"
#include "pose/pose_estimator.h"
#include "tracker/tracker.h"
#include <fstream>
#include <string>
#include <vector>

// Hardware-free post-detection pipeline: pose lookup → projection → JPDAF/Kalman → output.
// Shared by the live HeimdallApp and the offline replay driver.
class DetectionProcessor {
public:
    struct Config {
        std::vector<CameraParams> pose_cameras;
        ObjectTracker::Config     tracker;
        bool                      bypass_tracker = false;
        bool                      log_tracking   = false;
        std::string               log_path       = "logs/tracker_log.csv";
        size_t                    pose_buffer_capacity = PoseBuffer::N;
    };

    // output is owned externally (CommLayer in live mode, JsonlOutput in replay).
    DetectionProcessor(Config config, DetectionOutput& output);

    void push_pose(const CommLayer::TimestampedPose& p);
    void process(const std::vector<Detection>& dets);

private:
    Config          config_;
    DetectionOutput& output_;
    PoseEstimator   pose_estimator_;
    ObjectTracker   tracker_;
    PoseBuffer      pose_buffer_;
    std::ofstream   log_file_;
    std::ofstream   debug_log_file_;
    uint64_t        last_timestamp_ns_ = 0;
};
