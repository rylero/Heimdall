#pragma once
#include <cstdint>

enum class TrackEventType {
    CONFIRMED,
    UPDATED,
    LOST,
};

struct TrackedObject {
    uint32_t track_id;
    int      class_id;
    float    x, y;
    float    vx, vy;
    float    ax, ay;     // 0.f when model != CONSTANT_ACCELERATION
    float    confidence;
};

struct TrackEvent {
    TrackEventType type;
    TrackedObject  object;
};

// Kalman/JPDA internals from the most recent update cycle, for debug logging.
// gain_xx/gain_yy = p_xx/(p_xx+R), p_yy/(p_yy+R) -- diagonal Kalman gain.
struct TrackDebugInfo {
    uint32_t track_id;
    float    p_xx, p_yy;
    float    gain_xx, gain_yy;
    float    beta;
    float    maha;
};
