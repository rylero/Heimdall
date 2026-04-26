#pragma once
#include <array>
#include <cstdint>

enum class TrackStatus { TENTATIVE, CONFIRMED };

// Internal tracker state. Not exposed to callers — use TrackEvent / TrackedObject instead.
struct Track {
    uint32_t   id;
    int        class_id;
    float      confidence;

    // Kalman state: [x, y, vx, vy]  (meters, m/s, field-relative)
    std::array<float, 4>  state;
    // Covariance: 4x4 row-major
    std::array<float, 16> cov;

    double      last_update_s;
    int         frames_seen;    // total frames with associated detection (used for TENTATIVE->CONFIRMED promotion)
    int         frames_missed;  // consecutive frames with no associated detection
    TrackStatus status;
};
