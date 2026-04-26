#include "heimdall_app.h"
#include <chrono>

HeimdallApp::HeimdallApp(Config config)
    : config_(std::move(config)),
      pose_estimator_(config_.cameras),
      tracker_(config_.tracker),
      comm_(config_.comm),
      pipeline_(config_.cameras, config_.infer_config_path,
                [this](const std::vector<Detection>& d){ on_detections(d); })
{}

HeimdallApp::~HeimdallApp() { stop(); }

void HeimdallApp::on_detections(const std::vector<Detection>& dets) {
    RobotPose pose;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        pose = latest_pose_;
    }

    const auto field_dets = pose_estimator_.project(dets, pose);
    const uint64_t timestamp_ns = dets.empty() ? 0ULL : dets.front().timestamp_ns;

    const auto events = tracker_.update(field_dets,
        static_cast<double>(timestamp_ns) * 1e-9);

    comm_.send_frame(events, timestamp_ns, /*healthy=*/true);
}

void HeimdallApp::pose_recv_loop() {
    while (running_) {
        if (auto pose = comm_.try_recv_pose()) {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            latest_pose_ = *pose;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

void HeimdallApp::run() {
    running_ = true;
    pose_recv_thread_ = std::thread([this]{ pose_recv_loop(); });
    pipeline_.run();  // blocks until stop() or pipeline error
}

void HeimdallApp::stop() {
    running_ = false;
    pipeline_.stop();
    if (pose_recv_thread_.joinable())
        pose_recv_thread_.join();
}
