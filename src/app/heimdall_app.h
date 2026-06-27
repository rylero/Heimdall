#pragma once
#include "app/pose_buffer.h"
#include "apriltag/apriltag_detector.h"
#include "apriltag/tag_layout.h"
#include "comm/comm_layer.h"
#include "pipeline/camera_source.h"
#include "pipeline/pipeline.h"
#include "pose/pose_estimator.h"
#include "tracker/tracker.h"
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class HeimdallApp {
public:
    struct Config {
        // pipeline_cameras: device path + resolution for DeepStreamPipeline (CameraConfig)
        // pose_cameras:     intrinsics + extrinsics for PoseEstimator (CameraParams)
        // Both vectors must be the same length and in the same camera order.
        std::vector<CameraConfig>         pipeline_cameras;
        std::vector<CameraParams>         pose_cameras;
        std::string                       infer_config_path = "config/infer_yolo26n.txt";
        ObjectTracker::Config             tracker;
        CommLayer::Config                 comm;
        bool                              bypass_tracker = false;
        bool                              log_tracking   = false;
        std::string                       log_path       = "logs/tracker_log.csv";
        std::optional<std::string>        apriltag_layout_path; // nullopt = disabled
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
    std::ofstream      log_file_;
    std::ofstream      debug_log_file_;
    uint64_t           last_timestamp_ns_ = 0;
    std::atomic<bool>  running_{false};
    std::atomic<bool>  stopped_{false};
    std::thread        pose_recv_thread_;

    static constexpr int               kMaxDetQueue = 8;
    std::queue<std::vector<Detection>> det_queue_;
    std::mutex                         det_mutex_;
    std::condition_variable            det_cv_;
    std::thread                        det_worker_thread_;

    std::unique_ptr<AprilTagDetector> at_detector_;
    std::thread                       at_thread_;

    void on_detections(const std::vector<Detection>& dets);
    void enqueue_detections(const std::vector<Detection>& dets);
    void det_worker_loop();
    void pose_recv_loop();
    void apriltag_loop();
};
