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

    // 1. Unproject pixel to distorted normalised coords, then iteratively undistort.
    //    intr.cx/fx are already flip-adjusted, so (xd,yd) are in original camera space.
    const float xd = (px - intr.cx) / intr.fx;
    const float yd = (py - intr.cy) / intr.fy;
    float u = xd, v = yd;
    for (int i = 0; i < 5; ++i) {
        const float r2  = u*u + v*v;
        const float r4  = r2 * r2;
        const float r6  = r4 * r2;
        const float rad = 1.f + intr.k1*r2 + intr.k2*r4 + intr.k3*r6;
        u = (xd - 2.f*intr.p1*u*v - intr.p2*(r2 + 2.f*u*u)) / rad;
        v = (yd - intr.p1*(r2 + 2.f*v*v) - 2.f*intr.p2*u*v) / rad;
    }

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

        // Empirical quadratic distance correction: actual = a*d^2 + b*d + c
        // Calibrated from 5 measurements (1–3.4 m range). Recalibrate if mounting changes.
        const float raw_d = std::sqrt(fx*fx + fy*fy);
        if (raw_d > 0.01f) {
            constexpr float kA = -0.0658f;
            constexpr float kB =  0.9637f;
            constexpr float kC =  0.12f;
            const float corr_d = std::max(0.f, (kA * raw_d + kB) * raw_d + kC);
            const float scale  = corr_d / raw_d;
            fx *= scale;
            fy *= scale;
        }

        results.push_back({det.class_id, fx, fy, det.confidence});
    }

    return results;
}
