#pragma once
#include "app/detection_processor.h"
#include "app/recording.h"
#include "apriltag/apriltag_detector.h"
#include "apriltag/tag_layout.h"
#include "comm/comm_layer.h"
#include "pipeline/camera_source.h"
#include "pipeline/pipeline.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class HeimdallApp {
public:
    struct Config {
        std::vector<CameraConfig>         pipeline_cameras;
        std::vector<CameraParams>         pose_cameras;
        std::string                       infer_config_path = "config/infer_yolo26n.txt";
        ObjectTracker::Config             tracker;
        CommLayer::Config                 comm;
        bool                              bypass_tracker = false;
        bool                              log_tracking   = false;
        std::string                       log_path       = "logs/tracker_log.csv";
        std::optional<std::string>        apriltag_layout_path; // nullopt = disabled
        std::optional<std::string>        record_path;           // nullopt = recording disabled
    };

    explicit HeimdallApp(Config config);
    ~HeimdallApp();

    void run();
    void stop();

    // Trigger a one-shot debug snapshot of the next processed frame (inputs+outputs → JSON).
    void request_snapshot() { processor_.request_snapshot(); }

private:
    Config             config_;
    CommLayer          comm_;
    DeepStreamPipeline pipeline_;
    // Declared before processor_: its address is handed to the processor's Config
    // (healthy_flag), so it must be constructed first (member init order = declaration order).
    std::atomic<bool>  pipeline_healthy_{true};
    DetectionProcessor processor_;

    std::unique_ptr<RecordingWriter> recorder_;

    std::atomic<bool>  running_{false};
    std::atomic<bool>  stopped_{false};
    std::thread        pose_recv_thread_;
    std::thread        watchdog_thread_;

    static constexpr int               kMaxDetQueue = 8;
    std::queue<std::vector<Detection>> det_queue_;
    std::mutex                         det_mutex_;
    std::condition_variable            det_cv_;
    std::thread                        det_worker_thread_;

    std::unique_ptr<AprilTagDetector> at_detector_;
    std::thread                       at_thread_;

    void enqueue_detections(const std::vector<Detection>& dets);
    void det_worker_loop();
    void pose_recv_loop();
    void apriltag_loop();
    void watchdog_loop();
};
