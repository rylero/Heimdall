#pragma once
#include "track.h"

struct KalmanParams {
    float meas_noise_r    = 0.16f;  // measurement noise variance (m^2)
    float process_noise_q = 2.0f;   // process noise intensity
    float pos_cov_floor   = 0.1f;   // minimum position covariance diagonal
};

// Per-model predict (each uses its own stride N = 2, 4, or 6).
void kalman_predict_cp(Track& track, double dt, const KalmanParams& kp);
void kalman_predict_cv(Track& track, double dt, const KalmanParams& kp);
void kalman_predict_ca(Track& track, double dt, const KalmanParams& kp);

// Per-model update with combined JPDAF innovation.
void kalman_update_cp(Track& track, float innov_x, float innov_y, float total_weight, const KalmanParams& kp);
void kalman_update_cv(Track& track, float innov_x, float innov_y, float total_weight, const KalmanParams& kp);
void kalman_update_ca(Track& track, float innov_x, float innov_y, float total_weight, const KalmanParams& kp);

// Dispatchers — called by jpda.cpp; route to the correct model function via track.model.
void kalman_predict(Track& track, double dt, const KalmanParams& kp = {});
void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight, const KalmanParams& kp = {});

// Construct a new tentative track with the given filter model.
Track make_track(uint32_t id, int class_id, float x, float y,
                 float confidence, double timestamp_s, FilterModel model);
