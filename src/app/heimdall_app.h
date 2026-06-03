#pragma once
#include "app/pose_buffer.h"
#include "comm/comm_layer.h"
#include "pipeline/camera_source.h"
#include "pipeline/pipeline.h"
#include "pose/pose_estimator.h"
#include "tracker/tracker.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

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

    static constexpr int               kMaxDetQueue = 8;
    std::queue<std::vector<Detection>> det_queue_;
    std::mutex                         det_mutex_;
    std::condition_variable            det_cv_;
    std::thread                        det_worker_thread_;

    void on_detections(const std::vector<Detection>& dets);
    void enqueue_detections(const std::vector<Detection>& dets);
    void det_worker_loop();
    void pose_recv_loop();
};
