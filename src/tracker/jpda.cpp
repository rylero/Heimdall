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
    const KalmanParams kp{ cfg.meas_noise_r, cfg.process_noise_q, cfg.pos_cov_floor };
    for (auto& t : tracks) {
        double dt = timestamp_s - t.last_update_s;
        if (dt > 0.0) kalman_predict(t, dt, kp);
    }

    // Early-out: no tracks or no detections
    if (n == 0 || m == 0) {
        for (auto& t : tracks) {
            ++t.frames_missed;
            t.last_update_s = timestamp_s;
            t.dbg_beta = 0.f;
            t.dbg_maha = 0.f;
        }
        std::vector<int> all(m);
        for (int j = 0; j < m; ++j) all[j] = j;
        return all;
    }

    // 2. Compute Gaussian likelihoods within distance gate
    std::vector<std::vector<float>> L(n, std::vector<float>(m, 0.f));
    for (int i = 0; i < n; ++i) {
        const auto& P   = tracks[i].cov;
        const int   N   = (tracks[i].model == FilterModel::CONSTANT_POSITION)     ? 2 :
                          (tracks[i].model == FilterModel::CONSTANT_ACCELERATION)  ? 6 : 4;
        const float s00 = P[0*N+0] + cfg.meas_noise_r;
        const float s11 = P[1*N+1] + cfg.meas_noise_r;
        const float norm = 1.f / (TWO_PI * std::sqrt(s00 * s11));

        tracks[i].dbg_p_xx = P[0*N+0];
        tracks[i].dbg_p_yy = P[1*N+1];
        tracks[i].dbg_maha = 0.f;
        float min_maha = -1.f;

        for (int j = 0; j < m; ++j) {
            float dx = detections[j].x - tracks[i].state[0];
            float dy = detections[j].y - tracks[i].state[1];
            if (std::sqrt(dx*dx + dy*dy) >= cfg.gate_distance) continue;
            float maha = (dx*dx) / s00 + (dy*dy) / s11;
            if (min_maha < 0.f || maha < min_maha) min_maha = maha;
            if (maha > cfg.mahalanobis_gate) continue;  // statistically incompatible -- leave L=0
            L[i][j] = norm * std::exp(-0.5f * maha);
        }
        if (min_maha >= 0.f) tracks[i].dbg_maha = min_maha;
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
        float sum_bxx    = 0.f;   // Σ β_j νx_j²   — for the spread-of-innovations term
        float sum_byy    = 0.f;   // Σ β_j νy_j²
        float sum_bxy    = 0.f;   // Σ β_j νx_j νy_j

        for (int j = 0; j < m; ++j) {
            if (beta[i][j] <= 0.f) continue;
            const float nx = detections[j].x - tracks[i].state[0];
            const float ny = detections[j].y - tracks[i].state[1];
            innov_x    += beta[i][j] * nx;
            innov_y    += beta[i][j] * ny;
            sum_bxx    += beta[i][j] * nx * nx;
            sum_byy    += beta[i][j] * ny * ny;
            sum_bxy    += beta[i][j] * nx * ny;
            total_beta += beta[i][j];
        }

        tracks[i].dbg_beta = total_beta;

        // Spread of innovations M = Σ β_j ν_j ν_jᵀ − ν̄ ν̄ᵀ. Widens the posterior when the
        // track is torn between multiple gated detections — the term standard PDAF/JPDAF
        // includes and whose omission collapsed P to zero on long/ambiguous tracks (5.2).
        const float spread_xx = sum_bxx - innov_x * innov_x;
        const float spread_yy = sum_byy - innov_y * innov_y;
        const float spread_xy = sum_bxy - innov_x * innov_y;

        kalman_update_combined(tracks[i], innov_x, innov_y, total_beta, kp,
                               spread_xx, spread_xy, spread_yy);
        tracks[i].last_update_s = timestamp_s;

        if (total_beta > 0.f) {
            ++tracks[i].frames_seen;
            tracks[i].frames_missed = 0;
        } else {
            ++tracks[i].frames_missed;
        }
    }

    // 5. Spawn candidates. A detection is "claimed" if it is the single best-gated
    //    (max-likelihood) detection of at least one track. Everything else is a spawn
    //    candidate: detections outside every gate, AND detections that are gated but are
    //    no track's primary match — i.e. a second object sharing one track's gate.
    //    Fixes 5.19: the old rule (sum L == 0) only spawned fully-ungated detections, so
    //    two objects inside one gate lost a track until they separated beyond it.
    std::vector<char> claimed(m, 0);
    for (int i = 0; i < n; ++i) {
        int   best_j = -1;
        float best_L = 0.f;
        for (int j = 0; j < m; ++j)
            if (L[i][j] > best_L) { best_L = L[i][j]; best_j = j; }
        if (best_j >= 0) claimed[best_j] = 1;
    }

    // Duplicate suppression: an unclaimed detection that sits essentially on top of an existing
    // track (Mahalanobis^2 < dup_spawn_gate) is the SAME object seen again — most often a second
    // camera whose projection lands at the same field xy — not a new object. Without this, every
    // frame leaves that coincident detection unclaimed and spawns a fresh track that never dies
    // (a detection is always in its gate), so tracks grow unbounded on any multi-camera overlap.
    // Detections beyond dup_spawn_gate still spawn, so two genuinely distinct objects sharing one
    // association gate still split (preserves the 5.19 behaviour).
    std::vector<int> unassociated;
    const float dup_r2 = cfg.dup_spawn_radius * cfg.dup_spawn_radius;
    for (int j = 0; j < m; ++j) {
        if (claimed[j]) continue;
        if (cfg.dup_spawn_radius > 0.f) {
            bool duplicate = false;
            for (int i = 0; i < n; ++i) {
                const float dx = detections[j].x - tracks[i].state[0];
                const float dy = detections[j].y - tracks[i].state[1];
                if (dx*dx + dy*dy < dup_r2) { duplicate = true; break; }
            }
            if (duplicate) continue;  // coincident duplicate of an existing track — absorb
        }
        unassociated.push_back(j);
    }
    return unassociated;
}
