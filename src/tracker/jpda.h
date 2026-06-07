#pragma once
#include "field_detection.h"
#include "track.h"
#include <vector>

struct JpdaConfig {
    float gate_distance   = 1.0f;  // meters, coarse Euclidean pre-filter (cheap reject before maha calc)
    float clutter_density = 1.0f;  // lambda -- expected clutter returns per gated region
    float p_detection     = 0.9f;  // P_D -- probability of detecting a present target
};

// Chi-square critical value, 2 DOF, 99% confidence. Detections with Mahalanobis distance
// beyond this are statistically incompatible with the track and excluded (L=0) so they can
// be marked unassociated (spawn new track) rather than silently absorbed with beta~=0.
inline constexpr float MAHALANOBIS_GATE_THRESHOLD = 9.21f;

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
