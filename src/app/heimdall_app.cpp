#include "heimdall_app.h"
#include "apriltag/tag_layout.h"
#include <chrono>
#include <cstdio>

HeimdallApp::HeimdallApp(Config config)
    : config_(std::move(config)),
      pose_estimator_(config_.pose_cameras),
      tracker_(config_.tracker),
      comm_(config_.comm),
      pipeline_(config_.pipeline_cameras, config_.infer_config_path,
                [this](const std::vector<Detection>& d){ enqueue_detections(d); })
{
    if (config_.apriltag_layout_path) {
        try {
            AprilTagLayout layout = load_apriltag_layout(*config_.apriltag_layout_path);
            at_detector_ = std::make_unique<AprilTagDetector>(std::move(layout));
            if (!at_detector_->is_open()) {
                std::fprintf(stderr, "[apriltag] camera failed to open — AprilTag disabled\n");
                at_detector_.reset();
            } else {
                std::printf("[apriltag] detector ready\n");
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[apriltag] layout load failed: %s\n", e.what());
        }
    }
    if (config_.log_tracking) {
        log_file_.open(config_.log_path);
        log_file_ << "ts_s,source,track_id,x,y,conf\n";
        std::printf("[tracker log] writing to %s\n", config_.log_path.c_str());

        const std::string debug_path = config_.log_path + ".debug.csv";
        debug_log_file_.open(debug_path);
        debug_log_file_ << "ts_s,track_id,p_xx,p_yy,gain_xx,gain_yy,beta,maha\n";
        std::printf("[tracker log] writing debug info to %s\n", debug_path.c_str());
    }
}

HeimdallApp::~HeimdallApp() { stop(); }

void HeimdallApp::on_detections(const std::vector<Detection>& dets) {
    // Frames with zero detected objects carry no buf_pts here -- feeding the tracker a
    // fabricated timestamp of 0 corrupts every track's last_update_s, producing a huge
    // dt (and exploding covariance) on the next real-detection frame. Send a heartbeat
    // with the last known-good timestamp instead and skip tracker processing entirely.
    if (dets.empty()) {
        comm_.send_tracking_frame({}, last_timestamp_ns_, /*healthy=*/true);
        return;
    }

    const uint64_t timestamp_ns  = dets.front().timestamp_ns;
    const uint64_t capture_ns    = dets.front().capture_monotonic_ns;
    last_timestamp_ns_ = timestamp_ns;

    // Select the robot pose whose Jetson reception time is closest to the camera
    // capture time, compensating for DeepStream pipeline latency (typically 20–60 ms).
    const RobotPose pose = pose_buffer_.closest(capture_ns);

    const auto field_dets = pose_estimator_.project(dets, pose);
    const double ts_s = static_cast<double>(timestamp_ns) * 1e-9;

    if (config_.log_tracking && log_file_.is_open()) {
        for (const auto& fd : field_dets)
            log_file_ << ts_s << ",raw,-1," << fd.x << ',' << fd.y << ',' << fd.confidence << '\n';
    }

    // JPDAF tracker bypassed, just sends raw detections instead
    if (config_.bypass_tracker) {
        std::vector<TrackEvent> events;
        events.reserve(field_dets.size());
        for (int i = 0; i < static_cast<int>(field_dets.size()); ++i) {
            const auto& fd = field_dets[i];
            events.push_back({TrackEventType::UPDATED,
                TrackedObject{.track_id=static_cast<uint32_t>(i), .class_id=fd.class_id,
                              .x=fd.x, .y=fd.y, .confidence=fd.confidence}});
        }
        comm_.send_tracking_frame(events, timestamp_ns, /*healthy=*/true);
        return;
    }

    // update using the JPDAF tracker pipeline
    const auto events = tracker_.update(field_dets, ts_s);

    if (config_.log_tracking && log_file_.is_open()) {
        for (const auto& ev : events) {
            const char* src = (ev.type == TrackEventType::CONFIRMED) ? "confirmed"
                            : (ev.type == TrackEventType::LOST)      ? "lost"
                                                                      : "tracked";
            log_file_ << ts_s << ',' << src << ',' << ev.object.track_id
                      << ',' << ev.object.x << ',' << ev.object.y
                      << ',' << ev.object.confidence << '\n';
        }
    }

    if (config_.log_tracking && debug_log_file_.is_open()) {
        for (const auto& d : tracker_.debug_info()) {
            debug_log_file_ << ts_s << ',' << d.track_id
                            << ',' << d.p_xx << ',' << d.p_yy
                            << ',' << d.gain_xx << ',' << d.gain_yy
                            << ',' << d.beta << ',' << d.maha << '\n';
        }
    }

    comm_.send_tracking_frame(events, timestamp_ns, /*healthy=*/true);
}

void HeimdallApp::enqueue_detections(const std::vector<Detection>& dets) {
    std::lock_guard lock(det_mutex_);
    if (static_cast<int>(det_queue_.size()) >= kMaxDetQueue) return;  // drop if worker falls behind
    det_queue_.push(dets);
    det_cv_.notify_one();
}

void HeimdallApp::det_worker_loop() {
    while (true) {
        std::vector<Detection> dets;
        {
            std::unique_lock lock(det_mutex_);
            det_cv_.wait(lock, [this]{ return !det_queue_.empty() || !running_; });
            if (!running_ && det_queue_.empty()) break;
            dets = std::move(det_queue_.front());
            det_queue_.pop();
        }
        on_detections(dets);
    }
}

void HeimdallApp::pose_recv_loop() {
    while (running_) {
        if (auto p = comm_.try_recv_pose()) {
            if (at_detector_) {
                at_detector_->update_gyro(
                    static_cast<double>(p->pose.heading),
                    static_cast<double>(p->pose.vyaw),
                    p->jetson_recv_ns);
            }
            pose_buffer_.push(*p);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

void HeimdallApp::apriltag_loop() {
    while (running_) {
        auto result = at_detector_->detect();
        if (result) {
            comm_.send_apriltag_pose(
                static_cast<float>(result->x),
                static_cast<float>(result->y),
                static_cast<float>(result->heading_rad),
                result->timestamp_ns);
        }
    }
}

void HeimdallApp::run() {
    running_ = true;
    det_worker_thread_ = std::thread([this]{ det_worker_loop(); });
    pose_recv_thread_  = std::thread([this]{ pose_recv_loop(); });
    if (at_detector_) {
        at_thread_ = std::thread([this]{ apriltag_loop(); });
    }
    pipeline_.run();  // blocks until stop() or pipeline error
}

void HeimdallApp::stop() {
    if (stopped_.exchange(true)) return;  // guard against double-stop
    running_ = false;
    det_cv_.notify_all();
    pipeline_.stop();
    if (det_worker_thread_.joinable()) det_worker_thread_.join();
    if (pose_recv_thread_.joinable())  pose_recv_thread_.join();
    if (at_thread_.joinable())         at_thread_.join();
}
