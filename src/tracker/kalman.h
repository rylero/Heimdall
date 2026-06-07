#pragma once
#include "track.h"

// Measurement noise variance (m^2). ~0.2m dynamic pose error for a moving ball: 0.2^2 = 0.04
inline constexpr float MEAS_NOISE_R    = 0.04f;
// Process noise intensity. Higher = trust detections more, allow faster acceleration.
inline constexpr float PROCESS_NOISE_Q = 2.0f;
// Minimum position covariance diagonal. Prevents Kalman gain collapsing to ~0 on long tracks
// AND keeps per-frame correction (K * beta * gap) strong enough that a moving ball converges
// to a small steady-state lag rather than an equilibrium near the Mahalanobis gate edge.
// Floor=0.1 gives K >= 0.1/(0.1+0.04) ~= 0.71 (71% measurement weight minimum).
inline constexpr float POS_COV_FLOOR   = 0.1f;

// Per-model predict (each uses its own stride N = 2, 4, or 6).
void kalman_predict_cp(Track& track, double dt);
void kalman_predict_cv(Track& track, double dt);
void kalman_predict_ca(Track& track, double dt);

// Per-model update with combined JPDAF innovation.
void kalman_update_cp(Track& track, float innov_x, float innov_y, float total_weight);
void kalman_update_cv(Track& track, float innov_x, float innov_y, float total_weight);
void kalman_update_ca(Track& track, float innov_x, float innov_y, float total_weight);

// Dispatchers — called by jpda.cpp; route to the correct model function via track.model.
void kalman_predict(Track& track, double dt);
void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight);

// Construct a new tentative track with the given filter model.
Track make_track(uint32_t id, int class_id, float x, float y,
                 float confidence, double timestamp_s, FilterModel model);
