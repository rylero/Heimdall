#pragma once
#include "field_detection.h"
#include "jpda.h"
#include "track.h"
#include "track_event.h"
#include <vector>

struct ObjectTrackerConfig {
    int         confirmation_frames = 3;
    int         loss_frames         = 5;
    float       gate_distance       = 1.0f;
    float       clutter_density     = 1.0f;
    float       p_detection         = 0.9f;
    FilterModel filter_model        = FilterModel::CONSTANT_VELOCITY;
    float       meas_noise_r        = 0.16f;  // measurement noise variance (m^2)
    float       process_noise_q     = 2.0f;   // process noise intensity
    float       pos_cov_floor       = 0.1f;   // minimum position covariance diagonal
    float       mahalanobis_gate    = MAHALANOBIS_GATE_THRESHOLD;  // chi-square statistical gate (2 DOF)
    float       dup_spawn_radius    = 0.3f;   // meters; unclaimed det within this of a track is a duplicate, not a new track
};

class ObjectTracker {
public:
    using Config = ObjectTrackerConfig;

    explicit ObjectTracker(Config config = {});

    std::vector<TrackEvent> update(
        const std::vector<FieldDetection>& detections,
        double timestamp_s
    );

    // Snapshot of Kalman/JPDA internals from the most recent update() call, for debug logging.
    std::vector<TrackDebugInfo> debug_info() const;

private:
    Config             config_;
    std::vector<Track> tracks_;
    uint32_t           next_id_ = 1;

    TrackedObject to_tracked_object(const Track& t) const;
};
