#include "heimdall_app.h"
#include <chrono>

HeimdallApp::HeimdallApp(Config config)
    : config_(std::move(config)),
      pose_estimator_(config_.pose_cameras),
      tracker_(config_.tracker),
      comm_(config_.comm),
      pipeline_(config_.pipeline_cameras, config_.infer_config_path,
                [this](const std::vector<Detection>& d){ on_detections(d); })
{}

HeimdallApp::~HeimdallApp() { stop(); }

void HeimdallApp::on_detections(const std::vector<Detection>& dets) {
    const uint64_t timestamp_ns  = dets.empty() ? 0ULL : dets.front().timestamp_ns;
    const uint64_t capture_ns    = dets.empty() ? 0ULL : dets.front().capture_monotonic_ns;

    // Select the robot pose whose Jetson reception time is closest to the camera
    // capture time, compensating for DeepStream pipeline latency (typically 20–60 ms).
    const RobotPose pose = pose_buffer_.closest(capture_ns);

    // Publish raw pixel detections for web UI debug feed (before pose estimation)
    comm_.publish_raw(dets, timestamp_ns);

    const auto field_dets = pose_estimator_.project(dets, pose);
    const auto events = tracker_.update(field_dets,
        static_cast<double>(timestamp_ns) * 1e-9);

    comm_.send_frame(events, timestamp_ns, /*healthy=*/true);
}

void HeimdallApp::pose_recv_loop() {
    while (running_) {
        if (auto p = comm_.try_recv_pose()) {
            pose_buffer_.push(*p);
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
    if (stopped_.exchange(true)) return;  // guard against double-stop
    running_ = false;
    pipeline_.stop();
    if (pose_recv_thread_.joinable())
        pose_recv_thread_.join();
}
