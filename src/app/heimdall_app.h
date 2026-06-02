#pragma once
#include "comm/comm_layer.h"
#include "pipeline/camera_source.h"
#include "pipeline/pipeline.h"
#include "pose/pose_estimator.h"
#include "tracker/tracker.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// Ring buffer of recently received robot poses, indexed by Jetson CLOCK_MONOTONIC reception time.
// Allows on_detections() to select the pose that matches the camera frame's capture time
// rather than always using the latest (which is 20–60 ms newer than the captured image).
class PoseBuffer {
public:
    static constexpr size_t N = 64; // ~1.3 s of history at 50 Hz

    void push(const CommLayer::TimestampedPose& p) {
        std::lock_guard<std::mutex> lock(mu_);
        slots_[head_ % N] = p;
        ++head_;
        if (count_ < N) ++count_;
    }

    // Returns the pose whose jetson_recv_ns is closest to target_ns.
    // Returns a zero-initialized RobotPose if the buffer is empty (before first pose arrives).
    RobotPose closest(uint64_t target_ns) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (count_ == 0) return {};
        size_t best_idx  = (head_ - count_) % N;
        uint64_t best_diff = UINT64_MAX;
        for (size_t i = 0; i < count_; ++i) {
            const size_t idx   = (head_ - count_ + i) % N;
            const uint64_t t   = slots_[idx].jetson_recv_ns;
            const uint64_t diff = t >= target_ns ? t - target_ns : target_ns - t;
            if (diff < best_diff) { best_diff = diff; best_idx = idx; }
        }
        return slots_[best_idx].pose;
    }

private:
    mutable std::mutex mu_;
    std::array<CommLayer::TimestampedPose, N> slots_{};
    size_t   head_  = 0;
    size_t   count_ = 0;
};

class HeimdallApp {
public:
    struct Config {
        // pipeline_cameras: device path + resolution for DeepStreamPipeline (CameraConfig)
        // pose_cameras:     intrinsics + extrinsics for PoseEstimator (CameraParams)
        // Both vectors must be the same length and in the same camera order.
        std::vector<CameraConfig>  pipeline_cameras;
        std::vector<CameraParams>  pose_cameras;
        std::string                infer_config_path;
        ObjectTracker::Config      tracker;
        CommLayer::Config          comm;
    };

    explicit HeimdallApp(Config config);
    ~HeimdallApp();

    void run();    // blocks until stop()
    void stop();

private:
    Config             config_;
    PoseEstimator      pose_estimator_;
    ObjectTracker      tracker_;
    CommLayer          comm_;
    DeepStreamPipeline pipeline_;

    PoseBuffer         pose_buffer_;
    std::atomic<bool>  running_{false};
    std::atomic<bool>  stopped_{false};
    std::thread        pose_recv_thread_;

    void on_detections(const std::vector<Detection>& dets);
    void pose_recv_loop();
};
