#include "heimdall_app.h"
#include "apriltag/tag_layout.h"
#include <chrono>
#include <cstdio>

static DetectionProcessor::Config make_processor_config(const HeimdallApp::Config& c,
                                                        const std::atomic<bool>* health) {
    return {
        .pose_cameras          = c.pose_cameras,
        .tracker               = c.tracker,
        .bypass_tracker        = c.bypass_tracker,
        .log_tracking          = c.log_tracking,
        .log_path              = c.log_path,
        .pose_buffer_capacity  = PoseBuffer::N,
        .healthy_flag          = health,
    };
}

HeimdallApp::HeimdallApp(Config config)
    : config_(std::move(config)),
      comm_(config_.comm),
      pipeline_(config_.pipeline_cameras, config_.infer_config_path,
                [this](const std::vector<Detection>& d){ enqueue_detections(d); }),
      processor_(make_processor_config(config_, &pipeline_healthy_), comm_)
{
    if (config_.record_path)
        recorder_ = std::make_unique<RecordingWriter>(*config_.record_path);

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
}

HeimdallApp::~HeimdallApp() { stop(); }

void HeimdallApp::enqueue_detections(const std::vector<Detection>& dets) {
    std::lock_guard lock(det_mutex_);
    // Drop the OLDEST frame(s), never the newest (5.1). When the worker falls behind
    // (heavy JPDAF frame, snapshot write, log flush) the queue must shed stale backlog
    // and keep the freshest camera state — the "drop stale, never block" invariant. The
    // old tail-drop discarded every fresh frame while the worker chewed old ones, growing
    // latency without bound.
    while (static_cast<int>(det_queue_.size()) >= kMaxDetQueue)
        det_queue_.pop();
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
        if (recorder_) recorder_->write_frame(dets);
        processor_.process(dets);
    }
}

void HeimdallApp::pose_recv_loop() {
    while (running_) {
        if (auto p = comm_.try_recv_pose()) {
            if (recorder_) recorder_->write_pose(*p);
            if (at_detector_) {
                at_detector_->update_gyro(
                    static_cast<double>(p->pose.heading),
                    static_cast<double>(p->pose.vyaw),
                    p->jetson_recv_ns);
            }
            processor_.push_pose(*p);
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
                result->timestamp_ns,
                result->tag_count,
                static_cast<float>(result->avg_tag_distance),
                static_cast<float>(result->reproj_error),
                static_cast<float>(result->ambiguity),
                result->solve_mode);
        }
    }
}

void HeimdallApp::watchdog_loop() {
    // Silent-stall detection (5.16): the GStreamer bus posts nothing when a camera simply
    // stops delivering buffers, so watch the detection-probe frame counter. If it fails to
    // advance for kStallTimeout while we expect frames, flag unhealthy (so the robot's
    // isHealthy() sees it) and cycle the pipeline to recover. Restore health when frames
    // resume.
    using namespace std::chrono;
    constexpr auto kPoll        = milliseconds(500);
    constexpr auto kStallTimeout = seconds(3);

    uint64_t last_count = pipeline_frame_count();
    auto     last_progress = steady_clock::now();
    bool     restarted = false;

    while (running_) {
        std::this_thread::sleep_for(kPoll);
        const uint64_t count = pipeline_frame_count();
        const auto now = steady_clock::now();

        if (count != last_count) {
            last_count    = count;
            last_progress = now;
            if (!pipeline_healthy_.exchange(true))
                std::printf("[watchdog] pipeline recovered — healthy\n");
            restarted = false;
            continue;
        }

        // No progress. Only treat as a stall once we've ever seen a frame (count > 0),
        // to avoid false-flagging slow startup.
        if (count > 0 && now - last_progress >= kStallTimeout) {
            if (pipeline_healthy_.exchange(false))
                std::fprintf(stderr, "[watchdog] pipeline stalled — no frames for %llds, healthy=false\n",
                             (long long)duration_cast<seconds>(now - last_progress).count());
            if (!restarted) {
                pipeline_.restart();
                restarted = true;
            }
        }
    }
}

void HeimdallApp::run() {
    running_ = true;
    det_worker_thread_ = std::thread([this]{ det_worker_loop(); });
    pose_recv_thread_  = std::thread([this]{ pose_recv_loop(); });
    watchdog_thread_   = std::thread([this]{ watchdog_loop(); });
    if (at_detector_) {
        at_thread_ = std::thread([this]{ apriltag_loop(); });
    }
    pipeline_.run();
}

void HeimdallApp::stop() {
    if (stopped_.exchange(true)) return;
    running_ = false;
    det_cv_.notify_all();
    pipeline_.stop();
    if (det_worker_thread_.joinable()) det_worker_thread_.join();
    if (pose_recv_thread_.joinable())  pose_recv_thread_.join();
    if (watchdog_thread_.joinable())   watchdog_thread_.join();
    if (at_thread_.joinable())         at_thread_.join();
}
