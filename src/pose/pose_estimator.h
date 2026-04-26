#pragma once
#include "camera_params.h"
#include "pipeline/detection.h"
#include "tracker/field_detection.h"
#include <vector>

class PoseEstimator {
public:
    explicit PoseEstimator(std::vector<CameraParams> cameras);

    // Project camera-space detections to field-relative positions.
    // Detections whose ray does not intersect the ground are silently dropped.
    std::vector<FieldDetection> project(
        const std::vector<Detection>& detections,
        const RobotPose&              robot_pose
    ) const;

private:
    std::vector<CameraParams> cameras_;

    // Returns false if the ray does not intersect the ground plane (dfz >= 0).
    bool project_pixel(int camera_id,
                       float px, float py,
                       const RobotPose& robot_pose,
                       float& field_x, float& field_y) const;
};
