#pragma once
#include "track.h"

// Measurement noise variance (m^2). Derived from +/-7 cm pose accuracy: 0.07^2 ~= 0.005
inline constexpr float MEAS_NOISE_R    = 0.005f;
// Process noise intensity. Higher = trust detections more, allow faster acceleration.
inline constexpr float PROCESS_NOISE_Q = 0.1f;

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
