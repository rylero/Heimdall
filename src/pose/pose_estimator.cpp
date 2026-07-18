#include "pose_estimator.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// One-shot projection tracing. Set HEIMDALL_DEBUG_PROJ=1 to dump, for the first N
// detections, every intermediate of the pixel->field chain so the exact pixel space
// and each transform can be checked against a hand calculation. Off by default.
namespace {
bool proj_debug_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("HEIMDALL_DEBUG_PROJ");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}
int& proj_debug_budget() { static int n = 40; return n; }
}

PoseEstimator::PoseEstimator(std::vector<CameraParams> cameras)
    : cameras_(std::move(cameras)) {}

// Map a detection bbox (in the rotated frame nvinfer processed) to its ground-contact pixel
// (bottom-center of the object) in the native calibration frame. cam.rotation is the clockwise
// rotation the pipeline applied; W/H are the native frame dims. The four inverse maps below take
// a rotated pixel back to native coords. Since 0/90/180/270 send axis-aligned boxes to
// axis-aligned boxes, transforming the corners and taking native (mid-x, max-y) is exact.
static void ground_contact_native(const Detection& det, const CameraParams& cam,
                                   float& out_px, float& out_py) {
    const float W = static_cast<float>(cam.width);   // native frame dims
    const float H = static_cast<float>(cam.height);

    // Dims of the frame nvinfer actually saw (native, post-rotation): 90/270 swap W/H.
    const bool  swap = (cam.rotation == 90 || cam.rotation == 270);
    const float rotW = swap ? H : W;
    const float rotH = swap ? W : H;

    // Stage 1 — undo maintain-aspect letterbox from net space back to the rotated frame.
    // scale fits the rotated frame inside the net input; padding centers it. When infer dims
    // are unset (0), this is a no-op and det coords are taken as already-native.
    float scale = 1.f, padx = 0.f, pady = 0.f;
    if (cam.infer_width > 0 && cam.infer_height > 0) {
        const float nw = static_cast<float>(cam.infer_width);
        const float nh = static_cast<float>(cam.infer_height);
        scale = std::min(nw / rotW, nh / rotH);
        padx  = 0.5f * (nw - rotW * scale);
        pady  = 0.5f * (nh - rotH * scale);
    }

    const float xs[4] = {det.left, det.left + det.width, det.left,             det.left + det.width};
    const float ys[4] = {det.top,  det.top,              det.top + det.height, det.top + det.height};

    float minx = 1e30f, maxx = -1e30f, maxy = -1e30f;
    for (int i = 0; i < 4; ++i) {
        // net -> rotated-frame pixel
        const float rx = (xs[i] - padx) / scale;
        const float ry = (ys[i] - pady) / scale;
        // rotated-frame -> native (inverse of the clockwise pipeline rotation)
        float xn, yn;
        switch (cam.rotation) {
            case 90:  xn = ry;            yn = (H - 1.f) - rx; break;  // ccw-inverse of cw90
            case 180: xn = (W - 1.f) - rx; yn = (H - 1.f) - ry; break;
            case 270: xn = (W - 1.f) - ry; yn = rx;            break;
            default:  xn = rx;            yn = ry;             break;
        }
        minx = std::min(minx, xn);
        maxx = std::max(maxx, xn);
        maxy = std::max(maxy, yn);
    }
    out_px = 0.5f * (minx + maxx);  // bottom-center x in native frame
    out_py = maxy;                  // ground contact = lowest edge in native frame
}

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

    if (proj_debug_enabled() && proj_debug_budget() > 0) {
        std::fprintf(stderr,
            "[proj]   pixel=(%.1f,%.1f) -> undist(u,v)=(%.4f,%.4f) "
            "d_rob=(%.4f,%.4f,%.4f) d_field=(%.4f,%.4f,%.4f)\n"
            "[proj]   cam_origin_field=(%.4f,%.4f,%.4f) t=%.4f -> field=(%.4f,%.4f)\n",
            px, py, u, v, drx, dry, drz, dfx, dfy, dfz,
            ofx, ofy, ofz, t, field_x, field_y);
    }
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

        // Detections come from nvinfer, which ran on the rotated frame. Un-rotate the bbox
        // back into the native calibration frame (which the intrinsics describe), then take
        // bottom-center = ground contact there. Doing it here — rather than fudging the
        // intrinsics per-flip — keeps one clean transform and always picks the correct edge:
        // "bottom of the bbox in the rotated frame" is the object's head, not its feet, once
        // the image is rotated 180°.
        const auto& cam = cameras_[static_cast<size_t>(det.camera_id)];
        float px, py;
        ground_contact_native(det, cam, px, py);

        if (proj_debug_enabled() && proj_debug_budget() > 0) {
            std::fprintf(stderr,
                "[proj] cam=%d rot=%d native_WxH=%dx%d bbox(l,t,w,h)=(%.1f,%.1f,%.1f,%.1f) "
                "-> ground_contact_native=(%.1f,%.1f) | pose=(%.3f,%.3f,h=%.4f)\n",
                det.camera_id, cam.rotation, cam.width, cam.height,
                det.left, det.top, det.width, det.height, px, py,
                robot_pose.x, robot_pose.y, robot_pose.heading);
        }

        float fx, fy;
        if (!project_pixel(det.camera_id, px, py, robot_pose, fx, fy))
            continue;

        if (proj_debug_enabled() && proj_debug_budget() > 0)
            --proj_debug_budget();

        results.push_back({det.class_id, fx, fy, det.confidence});
    }

    return results;
}
