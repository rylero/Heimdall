#pragma once
#include "track.h"

// Measurement noise variance (m^2). Derived from +/-7 cm pose accuracy: 0.07^2 ~= 0.005
static constexpr float MEAS_NOISE_R    = 0.005f;
// Process noise intensity. Higher = trust detections more, allow faster acceleration.
static constexpr float PROCESS_NOISE_Q = 0.1f;

// Predict track forward by dt seconds (constant-velocity model).
void kalman_predict(Track& track, double dt);

// Update track with combined JPDAF innovation.
//   innov_x, innov_y = sum_j beta_ij * (z_j - H * x_pred)   (weighted innovation sum)
//   total_weight     = sum_j beta_ij = 1 - beta_i0
// No-ops when total_weight == 0 (no associated detections this frame).
void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight);

// Construct a new tentative track at (x,y) with zero velocity.
Track make_track(uint32_t id, int class_id, float x, float y, float confidence, double timestamp_s);
