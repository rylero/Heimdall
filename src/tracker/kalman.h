#pragma once
#include "track.h"

// Measurement noise variance (m^2). On-robot measurements showed jitter on a stationary
// target closer to ~0.4m dynamic pose error: 0.4^2 = 0.16. Was 0.04 (0.2m) -- too low,
// causing the filter to chase camera noise instead of smoothing it out.
inline constexpr float MEAS_NOISE_R    = 0.16f;
// Process noise intensity. Higher = trust detections more, allow faster acceleration.
inline constexpr float PROCESS_NOISE_Q = 2.0f;
// Minimum position covariance diagonal. Prevents Kalman gain collapsing to ~0 on long tracks
// AND keeps per-frame correction (K * beta * gap) strong enough that a moving ball converges
// to a small steady-state lag rather than an equilibrium near the Mahalanobis gate edge.
// Floor=0.1 gives K >= 0.1/(0.1+0.16) ~= 0.38 (38% measurement weight minimum).
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
