#pragma once
#include <cstdint>

enum class TrackEventType {
    CONFIRMED,  // track promoted from tentative to confirmed
    UPDATED,    // confirmed track updated with new position estimate
    LOST,       // confirmed track has not been seen for loss_frames -- retired
};

struct TrackedObject {
    uint32_t track_id;
    int      class_id;
    float    x, y;      // field-relative, meters (Kalman-filtered estimate)
    float    vx, vy;    // estimated velocity, m/s
    float    confidence;
};

struct TrackEvent {
    TrackEventType type;
    TrackedObject  object;
};
