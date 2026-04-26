#pragma once
#include "field_detection.h"
#include "jpda.h"
#include "track.h"
#include "track_event.h"
#include <vector>

class ObjectTracker {
public:
    struct Config {
        int   confirmation_frames = 3;    // frames_seen threshold: TENTATIVE -> CONFIRMED
        int   loss_frames         = 5;    // frames_missed threshold: CONFIRMED -> LOST
        float gate_distance       = 1.0f; // meters
        float clutter_density     = 1.0f; // JPDA lambda
        float p_detection         = 0.9f; // JPDA P_D
    };

    explicit ObjectTracker(Config config = {});

    // Process one frame. timestamp_s is monotonic seconds (e.g. buf_pts_ns / 1e9).
    // Returns events for this frame: CONFIRMED for newly confirmed tracks,
    // UPDATED for all active confirmed tracks, LOST for expired tracks.
    std::vector<TrackEvent> update(
        const std::vector<FieldDetection>& detections,
        double timestamp_s
    );

private:
    Config             config_;
    std::vector<Track> tracks_;
    uint32_t           next_id_ = 1;

    TrackedObject to_tracked_object(const Track& t) const;
};
