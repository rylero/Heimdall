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

    if (dets.empty()) {
        output_.send_tracking_frame({}, last_timestamp_ns_, /*healthy=*/true);
        if (take_snapshot) {
            snapshot::Frame snap;
            snap.frame_ts_ns = last_timestamp_ns_;
            snap.healthy     = true;
            snapshot::write(config_.snapshot_dir, snap);  // empty inputs — nothing detected this frame
        }
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
            write_snapshot(timestamp_ns, capture_ns, pose, dets, field_dets, events);
        output_.send_tracking_frame(events, timestamp_ns, /*healthy=*/true);
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
        write_snapshot(timestamp_ns, capture_ns, pose, dets, field_dets, events);

    output_.send_tracking_frame(events, timestamp_ns, /*healthy=*/true);
}

void DetectionProcessor::write_snapshot(uint64_t timestamp_ns, uint64_t capture_ns,
                                        const RobotPose& pose,
                                        const std::vector<Detection>& dets,
                                        const std::vector<FieldDetection>& field_dets,
                                        const std::vector<TrackEvent>& events) {
    snapshot::Frame snap;
    snap.frame_ts_ns          = timestamp_ns;
    snap.capture_monotonic_ns = capture_ns;
    snap.robot_pose           = pose;
    snap.detections           = dets;
    snap.field_detections     = field_dets;
    snap.events               = events;
    snap.healthy              = true;
    snapshot::write(config_.snapshot_dir, snap);
}
