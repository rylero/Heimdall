#include "app/detection_processor.h"
#include "app/snapshot.h"
#include <cstdio>

DetectionProcessor::DetectionProcessor(Config config, DetectionOutput& output)
    : config_(std::move(config)),
      output_(output),
      pose_estimator_(config_.pose_cameras),
      tracker_(config_.tracker),
      pose_buffer_(config_.pose_buffer_capacity)
{
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

void DetectionProcessor::push_pose(const CommLayer::TimestampedPose& p) {
    pose_buffer_.push(p);
}

void DetectionProcessor::process(const std::vector<Detection>& dets) {
    const bool take_snapshot = snapshot_requested_.exchange(false, std::memory_order_relaxed);
    // Pipeline health (5.16): forwarded on every tracking frame. Null flag → healthy (replay).
    const bool healthy = !config_.healthy_flag
                       || config_.healthy_flag->load(std::memory_order_relaxed);

    // No per-frame timestamp exists when dets is empty (the stamp rides on the Detections,
    // and there are none). Extrapolate from the last frame's timestamp using steady_clock
    // elapsed time so the tracker still sees real dt and can advance frames_missed / emit
    // LOST — skipping tracker_.update() here freezes every track's miss-streak forever.
    // last_timestamp_ns_ and wall_now are both steady_clock (the probe now stamps detections
    // with steady_clock::now(), see probe.cpp/5.14), so this blend is single-domain — no
    // clock-mixing (resolves 5.13).
    const auto wall_now = std::chrono::steady_clock::now();
    const double wall_elapsed_s = std::chrono::duration<double>(wall_now - last_wall_time_).count();
    last_wall_time_ = wall_now;

    if (dets.empty()) {
        const double ts_s = static_cast<double>(last_timestamp_ns_) * 1e-9 + wall_elapsed_s;
        const auto events = tracker_.update({}, ts_s);
        last_timestamp_ns_ = static_cast<uint64_t>(ts_s * 1e9);

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

        if (take_snapshot) {
            snapshot::Frame snap;
            snap.frame_ts_ns = last_timestamp_ns_;
            snap.events      = events;  // may carry LOST for tracks that timed out on empty frames
            snap.healthy     = healthy;
            snapshot::write(config_.snapshot_dir, snap);  // empty inputs — nothing detected this frame
        }
        output_.send_tracking_frame(events, last_timestamp_ns_, healthy);
        return;
    }

    const uint64_t timestamp_ns = dets.front().timestamp_ns;
    const uint64_t capture_ns   = dets.front().capture_monotonic_ns;
    last_timestamp_ns_ = timestamp_ns;

    const RobotPose pose      = pose_buffer_.closest(capture_ns);
    const auto      field_dets = pose_estimator_.project(dets, pose);
    const double    ts_s       = static_cast<double>(timestamp_ns) * 1e-9;

    if (config_.log_tracking && log_file_.is_open()) {
        for (const auto& fd : field_dets)
            log_file_ << ts_s << ",raw,-1," << fd.x << ',' << fd.y << ',' << fd.confidence << '\n';
    }

    if (config_.bypass_tracker) {
        std::vector<TrackEvent> events;
        events.reserve(field_dets.size());
        for (int i = 0; i < static_cast<int>(field_dets.size()); ++i) {
            const auto& fd = field_dets[i];
            events.push_back({TrackEventType::UPDATED,
                TrackedObject{.track_id=static_cast<uint32_t>(i), .class_id=fd.class_id,
                              .x=fd.x, .y=fd.y, .confidence=fd.confidence}});
        }
        if (take_snapshot)
            write_snapshot(timestamp_ns, capture_ns, pose, dets, field_dets, events, healthy);
        output_.send_tracking_frame(events, timestamp_ns, healthy);
        return;
    }

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

    if (take_snapshot)
        write_snapshot(timestamp_ns, capture_ns, pose, dets, field_dets, events, healthy);

    output_.send_tracking_frame(events, timestamp_ns, healthy);
}

void DetectionProcessor::write_snapshot(uint64_t timestamp_ns, uint64_t capture_ns,
                                        const RobotPose& pose,
                                        const std::vector<Detection>& dets,
                                        const std::vector<FieldDetection>& field_dets,
                                        const std::vector<TrackEvent>& events,
                                        bool healthy) {
    snapshot::Frame snap;
    snap.frame_ts_ns          = timestamp_ns;
    snap.capture_monotonic_ns = capture_ns;
    snap.robot_pose           = pose;
    snap.detections           = dets;
    snap.field_detections     = field_dets;
    snap.events               = events;
    snap.healthy              = healthy;
    snapshot::write(config_.snapshot_dir, snap);
}
