#pragma once
#include <array>
#include <cstdint>

enum class TrackStatus { TENTATIVE, CONFIRMED };

enum class FilterModel {
    CONSTANT_POSITION,      // state = [x, y],                N=2
    CONSTANT_VELOCITY,      // state = [x, y, vx, vy],        N=4
    CONSTANT_ACCELERATION,  // state = [x, y, vx, vy, ax, ay] N=6
};

// Internal tracker state. Not exposed to callers — use TrackEvent / TrackedObject instead.
struct Track {
    uint32_t    id;
    int         class_id;
    float       confidence;
    FilterModel model;

    // state[0..N-1] active; rest zero-initialised. N = 2, 4, or 6 per model.
    std::array<float, 6>  state;
    // cov uses row-major stride N×N occupying first N*N elements; rest zero.
    std::array<float, 36> cov;

    double      last_update_s;
    int         frames_seen;
    int         frames_missed;
    TrackStatus status;
};
