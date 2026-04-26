#include "jpda.h"
#include "kalman.h"
#include <cmath>
#include <vector>

static constexpr float TWO_PI = 6.28318530718f;

std::vector<int> jpda_update(
    std::vector<Track>&                tracks,
    const std::vector<FieldDetection>& detections,
    double                             timestamp_s,
    const JpdaConfig&                  cfg
) {
    const int n = static_cast<int>(tracks.size());
    const int m = static_cast<int>(detections.size());

    // 1. Predict all tracks forward
    for (auto& t : tracks) {
        double dt = timestamp_s - t.last_update_s;
        if (dt > 0.0) kalman_predict(t, dt);
    }

    // Early-out: no tracks or no detections
    if (n == 0 || m == 0) {
        for (auto& t : tracks) {
            ++t.frames_missed;
            t.last_update_s = timestamp_s;
        }
        std::vector<int> all(m);
        for (int j = 0; j < m; ++j) all[j] = j;
        return all;
    }

    // 2. Compute Gaussian likelihoods within Euclidean gate
    std::vector<std::vector<float>> L(n, std::vector<float>(m, 0.f));
    for (int i = 0; i < n; ++i) {
        const auto& P   = tracks[i].cov;
        const float s00 = P[0*4+0] + MEAS_NOISE_R;
        const float s11 = P[1*4+1] + MEAS_NOISE_R;
        const float norm = 1.f / (TWO_PI * std::sqrt(s00 * s11));

        for (int j = 0; j < m; ++j) {
            float dx = detections[j].x - tracks[i].state[0];
            float dy = detections[j].y - tracks[i].state[1];
            if (std::sqrt(dx*dx + dy*dy) >= cfg.gate_distance) continue;
            float maha = (dx*dx) / s00 + (dy*dy) / s11;
            L[i][j] = norm * std::exp(-0.5f * maha);
        }
    }

    // 3. Marginal JPDAF probabilities
    std::vector<std::vector<float>> beta(n, std::vector<float>(m, 0.f));
    for (int i = 0; i < n; ++i) {
        float sum_L = 0.f;
        for (int j = 0; j < m; ++j) sum_L += L[i][j];
        float denom = cfg.clutter_density + cfg.p_detection * sum_L;
        for (int j = 0; j < m; ++j)
            if (L[i][j] > 0.f)
                beta[i][j] = cfg.p_detection * L[i][j] / denom;
    }

    // 4. Update each track with combined weighted innovation
    for (int i = 0; i < n; ++i) {
        float innov_x    = 0.f;
        float innov_y    = 0.f;
        float total_beta = 0.f;

        for (int j = 0; j < m; ++j) {
            if (beta[i][j] <= 0.f) continue;
            innov_x    += beta[i][j] * (detections[j].x - tracks[i].state[0]);
            innov_y    += beta[i][j] * (detections[j].y - tracks[i].state[1]);
            total_beta += beta[i][j];
        }

        kalman_update_combined(tracks[i], innov_x, innov_y, total_beta);
        tracks[i].last_update_s = timestamp_s;

        if (total_beta > 0.f) {
            ++tracks[i].frames_seen;
            tracks[i].frames_missed = 0;
        } else {
            ++tracks[i].frames_missed;
        }
    }

    // 5. Unassociated detections: no track gated this detection (sum L == 0)
    std::vector<int> unassociated;
    for (int j = 0; j < m; ++j) {
        float total_L = 0.f;
        for (int i = 0; i < n; ++i) total_L += L[i][j];
        if (total_L == 0.f) unassociated.push_back(j);
    }
    return unassociated;
}
