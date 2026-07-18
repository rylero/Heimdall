#pragma once
#include "field_detection.h"
#include "track.h"
#include <vector>

// Chi-square critical value, 2 DOF, 99% confidence — default statistical gate. Detections
// with Mahalanobis distance beyond this are statistically incompatible with the track and
// excluded (L=0) so they can spawn a new track rather than be silently absorbed with beta~=0.
inline constexpr float MAHALANOBIS_GATE_THRESHOLD = 9.21f;

struct JpdaConfig {
    float gate_distance    = 1.0f;   // meters, coarse Euclidean pre-filter (cheap reject before maha calc)
    float clutter_density  = 1.0f;   // lambda -- expected clutter returns per gated region
    float p_detection      = 0.9f;   // P_D -- probability of detecting a present target
    float meas_noise_r     = 0.16f;  // measurement noise variance (m^2); passed through to Kalman
    float process_noise_q  = 2.0f;   // process noise intensity; passed through to Kalman
    float pos_cov_floor    = 0.1f;   // minimum position covariance diagonal; passed through to Kalman
    float mahalanobis_gate = MAHALANOBIS_GATE_THRESHOLD;  // chi-square gate (2 DOF); tune per clutter env
    // Duplicate-suppression radius (METERS, Euclidean). An unclaimed detection within this
    // distance of an existing track is treated as a coincident duplicate (e.g. the SAME object
    // seen by a second overlapping camera, which projects to the same field xy) and is NOT
    // spawned as a new track. Detections farther than this still spawn, so two genuinely distinct
    // objects sharing one association gate still split. Euclidean (not Mahalanobis) on purpose:
    // a young track's covariance is huge, so a Mahalanobis test would absorb distinct objects
    // near new tracks; a metric radius is cov-independent. Set 0 to disable.
    float dup_spawn_radius = 0.3f;
};

// Run one JPDAF cycle:
//   1. Predict all tracks to timestamp_s
//   2. Compute Gaussian likelihoods within Euclidean gate
//   3. Compute marginal beta association probabilities
//   4. Update each track with combined weighted innovation
//   5. Increment frames_seen / frames_missed counters
//
// Returns indices into `detections` that fell outside all track gates
// (no track gated this detection — candidates for new track creation).
std::vector<int> jpda_update(
    std::vector<Track>&                tracks,
    const std::vector<FieldDetection>& detections,
    double                             timestamp_s,
    const JpdaConfig&                  cfg
);
