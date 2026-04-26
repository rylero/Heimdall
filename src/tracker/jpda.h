#pragma once
#include "field_detection.h"
#include "track.h"
#include <vector>

struct JpdaConfig {
    float gate_distance   = 1.0f;  // meters, Euclidean gating radius
    float clutter_density = 1.0f;  // lambda -- expected clutter returns per gated region
    float p_detection     = 0.9f;  // P_D -- probability of detecting a present target
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
