#pragma once
#include "app/detection_output.h"
#include "app/pose_buffer.h"
#include "comm/comm_layer.h"
#include "pipeline/detection.h"
#include "pose/pose_estimator.h"
#include "tracker/tracker.h"
#include <atomic>
#include <chrono>
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
        std::string               snapshot_dir   = "logs/snapshots";
        size_t                    pose_buffer_capacity = PoseBuffer::N;
        // Optional pipeline-health signal (owned externally, e.g. by HeimdallApp's stall
        // watchdog). When null (replay/tests) health is reported healthy. When set, its
        // value is forwarded on every tracking frame so the robot's isHealthy() can see a
        // stall (5.16) instead of a hardcoded true.
        const std::atomic<bool>*  healthy_flag = nullptr;
    };

    // output is owned externally (CommLayer in live mode, JsonlOutput in replay).
    DetectionProcessor(Config config, DetectionOutput& output);

    void push_pose(const CommLayer::TimestampedPose& p);
    // frame_capture_ns: optional frame-level capture time. When nonzero it is used as the
    // timestamp for an EMPTY frame (replay feeds the recorded time so LOST/miss timing is
    // reproduced faithfully, 5.17). Zero (live) falls back to wall-clock extrapolation.
    void process(const std::vector<Detection>& dets, uint64_t frame_capture_ns = 0);

    // Request a one-shot debug snapshot: the next process() call dumps its full
    // inputs/intermediate/outputs to a JSON file. Thread-safe.
    void request_snapshot() { snapshot_requested_.store(true, std::memory_order_relaxed); }

private:
    void write_snapshot(uint64_t timestamp_ns, uint64_t capture_ns,
                        const RobotPose& pose,
                        const std::vector<Detection>& dets,
                        const std::vector<FieldDetection>& field_dets,
                        const std::vector<TrackEvent>& events,
                        bool healthy);

    Config          config_;
    DetectionOutput& output_;
    PoseEstimator   pose_estimator_;
    ObjectTracker   tracker_;
    PoseBuffer      pose_buffer_;
    std::ofstream   log_file_;
    std::ofstream   debug_log_file_;
    uint64_t        last_timestamp_ns_ = 0;
    std::chrono::steady_clock::time_point last_wall_time_ = std::chrono::steady_clock::now();
    std::atomic<bool> snapshot_requested_{false};
};
