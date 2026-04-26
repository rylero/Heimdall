#include "kalman.h"
#include <cmath>

void kalman_predict(Track& track, double dt_d) {
    const float dt = static_cast<float>(dt_d);
    auto& x = track.state;
    auto& P = track.cov;   // row-major: P[i*4+j]

    // Predict state: position += velocity * dt
    x[0] += x[2] * dt;
    x[1] += x[3] * dt;

    // Propagate covariance: P = F*P*F' + Q
    // F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
    //
    // Step 1: FP = F*P  (rows 0,1 of P mix in rows 2,3 via dt)
    std::array<float, 16> FP = P;
    for (int j = 0; j < 4; ++j) {
        FP[0*4+j] = P[0*4+j] + dt * P[2*4+j];
        FP[1*4+j] = P[1*4+j] + dt * P[3*4+j];
    }
    // Step 2: P = FP*F'  (cols 0,1 of FP mix in cols 2,3 via dt)
    for (int i = 0; i < 4; ++i) {
        P[i*4+0] = FP[i*4+0] + dt * FP[i*4+2];
        P[i*4+1] = FP[i*4+1] + dt * FP[i*4+3];
        P[i*4+2] = FP[i*4+2];
        P[i*4+3] = FP[i*4+3];
    }
    // Step 3: P += Q  (discrete white noise acceleration model)
    const float q    = PROCESS_NOISE_Q;
    const float dt2  = dt * dt;
    const float dt3  = dt2 * dt;
    const float dt4  = dt3 * dt;
    P[0*4+0] += q * dt4 / 4.f;
    P[1*4+1] += q * dt4 / 4.f;
    P[2*4+2] += q * dt2;
    P[3*4+3] += q * dt2;
    P[0*4+2] += q * dt3 / 2.f;   P[2*4+0] += q * dt3 / 2.f;
    P[1*4+3] += q * dt3 / 2.f;   P[3*4+1] += q * dt3 / 2.f;
}

void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight) {
    if (total_weight <= 0.f) return;

    auto& x = track.state;
    auto& P = track.cov;

    // S = H*P*H' + R  (2x2; H = [[1,0,0,0],[0,1,0,0]] selects top-left block of P)
    const float R   = MEAS_NOISE_R;
    const float s00 = P[0*4+0] + R,  s01 = P[0*4+1];
    const float s10 = P[1*4+0],       s11 = P[1*4+1] + R;
    const float det = s00 * s11 - s01 * s10;

    // S^-1  (analytical 2x2 inverse)
    const float si00 =  s11 / det,  si01 = -s01 / det;
    const float si10 = -s10 / det,  si11 =  s00 / det;

    // K = P*H'*S^-1  (4x2)
    // P*H' = first two columns of P
    float K[4][2];
    for (int i = 0; i < 4; ++i) {
        K[i][0] = P[i*4+0] * si00 + P[i*4+1] * si10;
        K[i][1] = P[i*4+0] * si01 + P[i*4+1] * si11;
    }

    // State update: x += K * innov
    x[0] += K[0][0] * innov_x + K[0][1] * innov_y;
    x[1] += K[1][0] * innov_x + K[1][1] * innov_y;
    x[2] += K[2][0] * innov_x + K[2][1] * innov_y;
    x[3] += K[3][0] * innov_x + K[3][1] * innov_y;

    // Covariance update: P = (I - w*K*H) * P
    // Save rows 0,1 before in-place modification (rows 2,3 unchanged by H)
    const float p0[4] = { P[0], P[1], P[2], P[3] };
    const float p1[4] = { P[4], P[5], P[6], P[7] };
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            P[i*4+j] -= total_weight * (K[i][0] * p0[j] + K[i][1] * p1[j]);
}

Track make_track(uint32_t id, int class_id, float x, float y, float confidence, double timestamp_s) {
    Track t{};
    t.id            = id;
    t.class_id      = class_id;
    t.confidence    = confidence;
    t.state         = { x, y, 0.f, 0.f };
    t.cov           = {
        10.f, 0.f,  0.f,  0.f,
        0.f,  10.f, 0.f,  0.f,
        0.f,  0.f,  1.f,  0.f,
        0.f,  0.f,  0.f,  1.f,
    };
    t.last_update_s = timestamp_s;
    t.frames_seen   = 1;
    t.frames_missed = 0;
    t.status        = TrackStatus::TENTATIVE;
    return t;
}
