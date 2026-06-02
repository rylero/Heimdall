#include "kalman.h"
#include <algorithm>
#include <cmath>

// ─── Constant-Position (CP) ───────────────────────────────────────────────────

void kalman_predict_cp(Track& track, double dt_d) {
    constexpr int N = 2;
    const float dt = static_cast<float>(dt_d);
    auto& P = track.cov;
    // F = I2 — position unchanged (random walk); P = P + q*dt*I2
    const float q = PROCESS_NOISE_Q;
    P[0*N+0] += q * dt;
    P[1*N+1] += q * dt;
}

void kalman_update_cp(Track& track, float innov_x, float innov_y, float total_weight) {
    if (total_weight <= 0.f) return;
    constexpr int N = 2;
    auto& x = track.state;
    auto& P = track.cov;

    const float R   = MEAS_NOISE_R;
    const float s00 = P[0*N+0] + R,  s01 = P[0*N+1];
    const float s10 = P[1*N+0],       s11 = P[1*N+1] + R;
    const float det = s00 * s11 - s01 * s10;

    const float si00 =  s11 / det,  si01 = -s01 / det;
    const float si10 = -s10 / det,  si11 =  s00 / det;

    float K[N][2];
    for (int i = 0; i < N; ++i) {
        K[i][0] = P[i*N+0] * si00 + P[i*N+1] * si10;
        K[i][1] = P[i*N+0] * si01 + P[i*N+1] * si11;
    }

    for (int i = 0; i < N; ++i)
        x[i] += K[i][0] * innov_x + K[i][1] * innov_y;

    float p0[N], p1[N];
    for (int j = 0; j < N; ++j) { p0[j] = P[0*N+j]; p1[j] = P[1*N+j]; }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            P[i*N+j] -= total_weight * (K[i][0] * p0[j] + K[i][1] * p1[j]);
}

// ─── Constant-Velocity (CV) — renamed from original kalman_predict/update ───

void kalman_predict_cv(Track& track, double dt_d) {
    constexpr int N = 4;
    const float dt = static_cast<float>(dt_d);
    auto& x = track.state;
    auto& P = track.cov;

    x[0] += x[2] * dt;
    x[1] += x[3] * dt;

    std::array<float, N*N> FP;
    std::copy_n(P.begin(), N*N, FP.begin());
    for (int j = 0; j < N; ++j) {
        FP[0*N+j] = P[0*N+j] + dt * P[2*N+j];
        FP[1*N+j] = P[1*N+j] + dt * P[3*N+j];
    }
    for (int i = 0; i < N; ++i) {
        P[i*N+0] = FP[i*N+0] + dt * FP[i*N+2];
        P[i*N+1] = FP[i*N+1] + dt * FP[i*N+3];
        P[i*N+2] = FP[i*N+2];
        P[i*N+3] = FP[i*N+3];
    }
    const float q   = PROCESS_NOISE_Q;
    const float dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
    P[0*N+0] += q * dt4 / 4.f;
    P[1*N+1] += q * dt4 / 4.f;
    P[2*N+2] += q * dt2;
    P[3*N+3] += q * dt2;
    P[0*N+2] += q * dt3 / 2.f;   P[2*N+0] += q * dt3 / 2.f;
    P[1*N+3] += q * dt3 / 2.f;   P[3*N+1] += q * dt3 / 2.f;
}

void kalman_update_cv(Track& track, float innov_x, float innov_y, float total_weight) {
    if (total_weight <= 0.f) return;
    constexpr int N = 4;
    auto& x = track.state;
    auto& P = track.cov;

    const float R   = MEAS_NOISE_R;
    const float s00 = P[0*N+0] + R,  s01 = P[0*N+1];
    const float s10 = P[1*N+0],       s11 = P[1*N+1] + R;
    const float det = s00 * s11 - s01 * s10;

    const float si00 =  s11 / det,  si01 = -s01 / det;
    const float si10 = -s10 / det,  si11 =  s00 / det;

    float K[N][2];
    for (int i = 0; i < N; ++i) {
        K[i][0] = P[i*N+0] * si00 + P[i*N+1] * si10;
        K[i][1] = P[i*N+0] * si01 + P[i*N+1] * si11;
    }

    for (int i = 0; i < N; ++i)
        x[i] += K[i][0] * innov_x + K[i][1] * innov_y;

    const float p0[N] = { P[0*N+0], P[0*N+1], P[0*N+2], P[0*N+3] };
    const float p1[N] = { P[1*N+0], P[1*N+1], P[1*N+2], P[1*N+3] };
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            P[i*N+j] -= total_weight * (K[i][0] * p0[j] + K[i][1] * p1[j]);
}

// ─── Constant-Acceleration (CA) — stubs replaced by real impl in a later task ─

void kalman_predict_ca(Track& /*track*/, double /*dt*/) {}

void kalman_update_ca(Track& /*track*/, float /*innov_x*/, float /*innov_y*/, float /*w*/) {}

// ─── Dispatchers ─────────────────────────────────────────────────────────────

void kalman_predict(Track& track, double dt) {
    switch (track.model) {
        case FilterModel::CONSTANT_POSITION:     kalman_predict_cp(track, dt); break;
        case FilterModel::CONSTANT_VELOCITY:     kalman_predict_cv(track, dt); break;
        case FilterModel::CONSTANT_ACCELERATION: kalman_predict_ca(track, dt); break;
    }
}

void kalman_update_combined(Track& track, float innov_x, float innov_y, float total_weight) {
    switch (track.model) {
        case FilterModel::CONSTANT_POSITION:     kalman_update_cp(track, innov_x, innov_y, total_weight); break;
        case FilterModel::CONSTANT_VELOCITY:     kalman_update_cv(track, innov_x, innov_y, total_weight); break;
        case FilterModel::CONSTANT_ACCELERATION: kalman_update_ca(track, innov_x, innov_y, total_weight); break;
    }
}

// ─── Track construction ───────────────────────────────────────────────────────

Track make_track(uint32_t id, int class_id, float x, float y, float confidence,
                 double timestamp_s, FilterModel model) {
    Track t{};
    t.id            = id;
    t.class_id      = class_id;
    t.confidence    = confidence;
    t.model         = model;
    t.state         = {};
    t.state[0]      = x;
    t.state[1]      = y;
    t.cov           = {};

    switch (model) {
        case FilterModel::CONSTANT_POSITION:
            t.cov[0*2+0] = 10.f;
            t.cov[1*2+1] = 10.f;
            break;
        case FilterModel::CONSTANT_VELOCITY:
            t.cov[0*4+0] = 10.f;
            t.cov[1*4+1] = 10.f;
            t.cov[2*4+2] = 1.f;
            t.cov[3*4+3] = 1.f;
            break;
        case FilterModel::CONSTANT_ACCELERATION:
            t.cov[0*6+0] = 10.f;
            t.cov[1*6+1] = 10.f;
            t.cov[2*6+2] = 1.f;
            t.cov[3*6+3] = 1.f;
            t.cov[4*6+4] = 0.1f;
            t.cov[5*6+5] = 0.1f;
            break;
    }

    t.last_update_s = timestamp_s;
    t.frames_seen   = 1;
    t.frames_missed = 0;
    t.status        = TrackStatus::TENTATIVE;
    return t;
}
