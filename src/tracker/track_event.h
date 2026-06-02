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
