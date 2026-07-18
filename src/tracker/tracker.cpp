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
        .meas_noise_r    = config_.meas_noise_r,
        .process_noise_q = config_.process_noise_q,
        .pos_cov_floor   = config_.pos_cov_floor,
        .mahalanobis_gate = config_.mahalanobis_gate,
        .dup_spawn_radius = config_.dup_spawn_radius,
    };

    auto unassociated = jpda_update(tracks_, detections, timestamp_s, jpda_cfg);

    // any tracks not associated by jpda are potential new tracks, so we will create them
    for (int idx : unassociated) {
        const auto& d = detections[idx];
        tracks_.push_back(make_track(next_id_++, d.class_id, d.x, d.y, d.confidence,
                                     timestamp_s, config_.filter_model));
    }

    std::vector<TrackEvent> events;

    // Merge coincident tracks. Two tracks of the same class that have converged onto the same
    // field position are the same physical object -- most often one object seen by two overlapping
    // cameras, which each spawned a track on the first frame (before either existed for
    // dup_spawn_radius to suppress the other) and then drifted together. dup_spawn_radius only
    // blocks NEW spawns; it can't collapse two tracks that already exist, so without this pass
    // such a pair persists forever and the object reports two coincident tracks. Keep the older
    // (lower id, encountered first since tracks_ stays id-ascending) and fold the younger into it.
    if (config_.dup_spawn_radius > 0.f) {
        const float r2 = config_.dup_spawn_radius * config_.dup_spawn_radius;
        std::vector<Track> kept;
        kept.reserve(tracks_.size());
        for (auto& t : tracks_) {
            bool merged = false;
            for (auto& k : kept) {
                if (k.class_id != t.class_id) continue;
                const float dx = k.state[0] - t.state[0];
                const float dy = k.state[1] - t.state[1];
                if (dx*dx + dy*dy < r2) {
                    k.confidence    = std::max(k.confidence, t.confidence);
                    k.frames_seen   = std::max(k.frames_seen, t.frames_seen);
                    k.frames_missed = std::min(k.frames_missed, t.frames_missed);
                    if (t.status == TrackStatus::CONFIRMED) {
                        k.status = TrackStatus::CONFIRMED;
                        // t was already reported to consumers -- retire its id explicitly.
                        events.push_back({TrackEventType::LOST, to_tracked_object(t)});
                    }
                    merged = true;
                    break;
                }
            }
            if (!merged) kept.push_back(std::move(t));
        }
        tracks_ = std::move(kept);
    }

    std::vector<Track> surviving;

    // go through an update tracks be either confirming, discarding, or updating them
    for (auto& t : tracks_) {
        if (t.status == TrackStatus::TENTATIVE) {
            if (t.frames_missed > 0) { // if we miss frames on a new track, it is likely noise
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

std::vector<TrackDebugInfo> ObjectTracker::debug_info() const {
    std::vector<TrackDebugInfo> out;
    out.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        out.push_back(TrackDebugInfo{
            t.id,
            t.dbg_p_xx, t.dbg_p_yy,
            t.dbg_p_xx / (t.dbg_p_xx + config_.meas_noise_r),
            t.dbg_p_yy / (t.dbg_p_yy + config_.meas_noise_r),
            t.dbg_beta,
            t.dbg_maha,
        });
    }
    return out;
}
