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
