#pragma once
#include "pipeline/camera_source.h"
#include "pose/camera_params.h"
#include <string>
#include <vector>

struct CameraLoadResult {
    std::vector<CameraConfig> pipeline_cameras;
    std::vector<CameraParams> pose_cameras;
};

// Load all *.json files from `dir`, sorted by filename.
// Each file describes one camera (pipeline + pose config).
CameraLoadResult load_camera_configs(const std::string& dir);
