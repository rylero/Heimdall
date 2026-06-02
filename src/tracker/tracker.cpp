#include "tracker.h"
#include "kalman.h"
#include <algorithm>

ObjectTracker::ObjectTracker(Config config) : config_(config) {}

TrackedObject ObjectTracker::to_tracked_object(const Track& t) const {
    const float vx = (t.model != FilterModel::CONSTANT_POSITION) ? t.state[2] : 0.f;
    const float vy = (t.model != FilterModel::CONSTANT_POSITION) ? t.state[3] : 0.f;
    const float ax = (t.model == FilterModel::CONSTANT_ACCELERATION) ? t.state[4] : 0.f;
    const float ay = (t.model == FilterModel::CONSTANT_ACCELERATION) ? t.state[5] : 0.f;
    return {
        .track_id   = t.id,
        .class_id   = t.class_id,
        .x          = t.state[0],
        .y          = t.state[1],
        .vx         = vx,
        .vy         = vy,
        .ax         = ax,
        .ay         = ay,
        .confidence = t.confidence,
    };
}

std::vector<TrackEvent> ObjectTracker::update(
    const std::vector<FieldDetection>& detections,
    double timestamp_s
) {
    JpdaConfig jpda_cfg{
        .gate_distance   = config_.gate_distance,
        .clutter_density = config_.clutter_density,
        .p_detection     = config_.p_detection,
    };

    auto unassociated = jpda_update(tracks_, detections, timestamp_s, jpda_cfg);

    for (int idx : unassociated) {
        const auto& d = detections[idx];
        tracks_.push_back(make_track(next_id_++, d.class_id, d.x, d.y, d.confidence,
                                     timestamp_s, config_.filter_model));
    }

    std::vector<TrackEvent> events;
    std::vector<Track> surviving;

    for (auto& t : tracks_) {
        if (t.status == TrackStatus::TENTATIVE) {
            if (t.frames_missed > 0) {
                continue;
            }
            if (t.frames_seen >= config_.confirmation_frames) {
                t.status = TrackStatus::CONFIRMED;
                events.push_back({TrackEventType::CONFIRMED, to_tracked_object(t)});
            }
            surviving.push_back(std::move(t));
            continue;
        }

        if (t.frames_missed >= config_.loss_frames) {
            events.push_back({TrackEventType::LOST, to_tracked_object(t)});
            continue;
        }

        events.push_back({TrackEventType::UPDATED, to_tracked_object(t)});
        surviving.push_back(std::move(t));
    }

    tracks_ = std::move(surviving);
    return events;
}
