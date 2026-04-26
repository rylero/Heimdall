#include "pose_estimator.h"
#include <cmath>

PoseEstimator::PoseEstimator(std::vector<CameraParams> cameras)
    : cameras_(std::move(cameras)) {}

bool PoseEstimator::project_pixel(int camera_id,
                                   float px, float py,
                                   const RobotPose& rp,
                                   float& field_x, float& field_y) const {
    const auto& cam  = cameras_[static_cast<size_t>(camera_id)];
    const auto& intr = cam.intrinsics;
    const auto& extr = cam.extrinsics;
    const auto& R    = extr.R;  // camera->robot, row-major 3x3

    // 1. Unproject pixel to direction in camera frame: d_cam = [u, v, 1]
    const float u = (px - intr.cx) / intr.fx;
    const float v = (py - intr.cy) / intr.fy;

    // 2. Rotate direction to robot frame: d_rob = R * d_cam
    const float drx = R[0]*u + R[1]*v + R[2];
    const float dry = R[3]*u + R[4]*v + R[5];
    const float drz = R[6]*u + R[7]*v + R[8];

    // 3. Rotate direction to field frame via robot heading
    const float ch = std::cos(rp.heading), sh = std::sin(rp.heading);
    const float dfx = ch * drx - sh * dry;
    const float dfy = sh * drx + ch * dry;
    const float dfz = drz;

    // Ray must point downward to intersect ground (z=0 plane)
    if (dfz >= 0.f) return false;

    // 4. Camera origin in field frame
    const float ofx = rp.x + ch * extr.tx - sh * extr.ty;
    const float ofy = rp.y + sh * extr.tx + ch * extr.ty;
    const float ofz = extr.tz;

    // 5. Ground intersection: ofz + t * dfz = 0
    const float t = -ofz / dfz;
    field_x = ofx + t * dfx;
    field_y = ofy + t * dfy;
    return true;
}

std::vector<FieldDetection> PoseEstimator::project(
    const std::vector<Detection>& detections,
    const RobotPose&              robot_pose
) const {
    std::vector<FieldDetection> results;
    results.reserve(detections.size());

    for (const auto& det : detections) {
        if (det.camera_id < 0 || det.camera_id >= static_cast<int>(cameras_.size()))
            continue;

        // Bottom-center of bounding box = ground contact point
        const float px = det.left + det.width  / 2.f;
        const float py = det.top  + det.height;

        float fx, fy;
        if (!project_pixel(det.camera_id, px, py, robot_pose, fx, fy))
            continue;

        results.push_back({det.class_id, fx, fy, det.confidence});
    }

    return results;
}
