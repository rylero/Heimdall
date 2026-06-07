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

    // Debug snapshot from the most recent jpda_update cycle (post-predict, pre-update).
    // p_xx/p_yy + MEAS_NOISE_R fully determine the diagonal Kalman gain K = P/(P+R).
    float       dbg_p_xx = 0.f;
    float       dbg_p_yy = 0.f;
    float       dbg_beta = 0.f;   // combined JPDAF association weight (0 = no detection associated)
    float       dbg_maha = 0.f;   // Mahalanobis distance^2 to closest-matching detection (0 if none gated)
};
