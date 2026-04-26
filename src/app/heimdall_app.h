#pragma once
#include "comm/comm_layer.h"
#include "pipeline/pipeline.h"
#include "pose/pose_estimator.h"
#include "tracker/tracker.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

class HeimdallApp {
public:
    struct Config {
        std::vector<CameraParams>  cameras;
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

    RobotPose          latest_pose_{};
    std::mutex         pose_mutex_;
    std::atomic<bool>  running_{false};
    std::thread        pose_recv_thread_;

    void on_detections(const std::vector<Detection>& dets);
    void pose_recv_loop();
};
