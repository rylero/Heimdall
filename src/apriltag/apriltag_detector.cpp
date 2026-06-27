#include "apriltag_detector.h"
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <deque>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------
// 4×4 homogeneous transform helpers (column-major OpenCV Mat, double)
// ---------------------------------------------------------------------------

static cv::Mat make_transform(double tx, double ty, double tz,
                               double roll, double pitch, double yaw) {
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
        1,          0,           0,
        0,  std::cos(roll), -std::sin(roll),
        0,  std::sin(roll),  std::cos(roll));
    cv::Mat R_y = (cv::Mat_<double>(3,3) <<
         std::cos(pitch), 0, std::sin(pitch),
         0,               1, 0,
        -std::sin(pitch), 0, std::cos(pitch));
    cv::Mat R_z = (cv::Mat_<double>(3,3) <<
        std::cos(yaw), -std::sin(yaw), 0,
        std::sin(yaw),  std::cos(yaw), 0,
        0,              0,             1);
    cv::Mat R = R_z * R_y * R_x;
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    T.at<double>(0,3) = tx;
    T.at<double>(1,3) = ty;
    T.at<double>(2,3) = tz;
    return T;
}

static cv::Mat invert_transform(const cv::Mat& T) {
    cv::Mat R     = T(cv::Rect(0,0,3,3)).t();
    cv::Mat t     = T(cv::Rect(3,0,1,3));
    cv::Mat t_inv = -R * t;
    cv::Mat T_inv = cv::Mat::eye(4,4,CV_64F);
    R.copyTo(T_inv(cv::Rect(0,0,3,3)));
    t_inv.copyTo(T_inv(cv::Rect(3,0,1,3)));
    return T_inv;
}

// ---------------------------------------------------------------------------
// Gyro pose history for timestamp interpolation
// ---------------------------------------------------------------------------

struct PoseSample {
    uint64_t timestamp_ns;
    double   yaw;   // radians, field-relative CCW positive
    double   vyaw;  // rad/s, for forward extrapolation
};

static constexpr size_t   POSE_HISTORY_SIZE = 60;         // ~1.2 s at 50 Hz
static constexpr uint64_t GYRO_STALE_NS     = 500'000'000ULL; // 500 ms

// ---------------------------------------------------------------------------
// V4L2 buffer count
// ---------------------------------------------------------------------------
static constexpr int V4L2_NUM_BUFS = 2;

// ---------------------------------------------------------------------------
// Pimpl — V4L2 camera + constrained solver state
// ---------------------------------------------------------------------------

struct AprilTagDetector::Impl {
    AprilTagLayout       layout;
    int                  v4l2_fd = -1;
    void*                buf_start[V4L2_NUM_BUFS]  = {};
    size_t               buf_length[V4L2_NUM_BUFS] = {};
    std::vector<uint8_t> graybuf;
    apriltag_family_t*   tf  = nullptr;
    apriltag_detector_t* det = nullptr;
    cv::Mat              cam_mat;
    cv::Mat              dist;
    cv::Mat              T_robot_cam;

    std::deque<PoseSample> pose_history;
    std::mutex             pose_mutex;

    Impl(AprilTagLayout lay) : layout(std::move(lay)) {
        auto& c = layout.camera;

        v4l2_fd = open(c.device.c_str(), O_RDWR);
        if (v4l2_fd < 0) {
            std::fprintf(stderr, "[apriltag] open %s: %s\n",
                         c.device.c_str(), std::strerror(errno));
            return;
        }

        // Request YUYV format — Y bytes are grayscale, no decoder needed
        struct v4l2_format fmt = {};
        fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width       = static_cast<__u32>(c.width);
        fmt.fmt.pix.height      = static_cast<__u32>(c.height);
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field       = V4L2_FIELD_NONE;
        if (ioctl(v4l2_fd, VIDIOC_S_FMT, &fmt) < 0) {
            std::fprintf(stderr, "[apriltag] VIDIOC_S_FMT: %s\n", std::strerror(errno));
            ::close(v4l2_fd); v4l2_fd = -1; return;
        }

        // Set frame rate (best-effort, camera may ignore)
        struct v4l2_streamparm parm = {};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator   = 1;
        parm.parm.capture.timeperframe.denominator =
            static_cast<__u32>(c.fps > 0 ? c.fps : 10);
        ioctl(v4l2_fd, VIDIOC_S_PARM, &parm);

        // Allocate mmap buffers
        struct v4l2_requestbuffers req = {};
        req.count  = V4L2_NUM_BUFS;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (ioctl(v4l2_fd, VIDIOC_REQBUFS, &req) < 0) {
            std::fprintf(stderr, "[apriltag] VIDIOC_REQBUFS: %s\n", std::strerror(errno));
            ::close(v4l2_fd); v4l2_fd = -1; return;
        }

        for (int i = 0; i < V4L2_NUM_BUFS; ++i) {
            struct v4l2_buffer buf = {};
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = static_cast<__u32>(i);
            if (ioctl(v4l2_fd, VIDIOC_QUERYBUF, &buf) < 0) {
                std::fprintf(stderr, "[apriltag] VIDIOC_QUERYBUF %d: %s\n",
                             i, std::strerror(errno));
                ::close(v4l2_fd); v4l2_fd = -1; return;
            }
            buf_length[i] = buf.length;
            buf_start[i]  = mmap(nullptr, buf.length,
                                 PROT_READ | PROT_WRITE, MAP_SHARED,
                                 v4l2_fd, buf.m.offset);
            if (buf_start[i] == MAP_FAILED) {
                buf_start[i] = nullptr;
                std::fprintf(stderr, "[apriltag] mmap buf %d: %s\n",
                             i, std::strerror(errno));
                ::close(v4l2_fd); v4l2_fd = -1; return;
            }
            struct v4l2_buffer qbuf = buf;
            if (ioctl(v4l2_fd, VIDIOC_QBUF, &qbuf) < 0) {
                std::fprintf(stderr, "[apriltag] VIDIOC_QBUF %d: %s\n",
                             i, std::strerror(errno));
                ::close(v4l2_fd); v4l2_fd = -1; return;
            }
        }

        enum v4l2_buf_type btype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(v4l2_fd, VIDIOC_STREAMON, &btype) < 0) {
            std::fprintf(stderr, "[apriltag] VIDIOC_STREAMON: %s\n", std::strerror(errno));
            for (int i = 0; i < V4L2_NUM_BUFS; ++i)
                if (buf_start[i]) munmap(buf_start[i], buf_length[i]);
            ::close(v4l2_fd); v4l2_fd = -1; return;
        }

        graybuf.resize(static_cast<size_t>(c.width) * static_cast<size_t>(c.height));

        tf  = tag36h11_create();
        det = apriltag_detector_create();
        apriltag_detector_add_family(det, tf);
        det->quad_decimate = 2.0f;
        det->nthreads      = 2;
        det->debug         = 0;
        det->refine_edges  = 1;

        cam_mat = (cv::Mat_<double>(3,3) <<
            c.fx, 0,    c.cx,
            0,    c.fy, c.cy,
            0,    0,    1);
        dist = (cv::Mat_<double>(1,5) << c.k1, c.k2, c.p1, c.p2, c.k3);

        auto& r = layout.robot_to_camera;
        T_robot_cam = make_transform(r.x, r.y, r.z, r.roll, r.pitch, r.yaw);
    }

    ~Impl() {
        if (det) apriltag_detector_destroy(det);
        if (tf)  tag36h11_destroy(tf);
        if (v4l2_fd >= 0) {
            enum v4l2_buf_type btype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(v4l2_fd, VIDIOC_STREAMOFF, &btype);
            for (int i = 0; i < V4L2_NUM_BUFS; ++i)
                if (buf_start[i]) munmap(buf_start[i], buf_length[i]);
            ::close(v4l2_fd);
        }
    }

    // Interpolate (or extrapolate) gyro yaw to target_ns using the stored history.
    // Uses Jetson CLOCK_MONOTONIC timestamps — same domain as V4L2 buffer timestamps.
    std::optional<double> interpolate_yaw(uint64_t target_ns) {
        std::lock_guard<std::mutex> lock(pose_mutex);
        if (pose_history.empty()) return std::nullopt;

        // Reject stale gyro stream (robot disconnected or hasn't sent pose yet)
        if (pose_history.back().timestamp_ns + GYRO_STALE_NS < target_ns)
            return std::nullopt;

        // Extrapolate forward from most-recent sample (common case at ~50 Hz)
        const auto& back = pose_history.back();
        if (target_ns >= back.timestamp_ns) {
            double dt = static_cast<double>(
                static_cast<int64_t>(target_ns - back.timestamp_ns)) * 1e-9;
            return back.yaw + back.vyaw * dt;
        }

        // Find first sample with timestamp >= target_ns
        auto it = std::lower_bound(pose_history.begin(), pose_history.end(), target_ns,
            [](const PoseSample& s, uint64_t t) { return s.timestamp_ns < t; });

        if (it == pose_history.begin()) {
            // target is before history — extrapolate backward from oldest sample
            double dt = static_cast<double>(
                static_cast<int64_t>(target_ns - it->timestamp_ns)) * 1e-9;
            return it->yaw + it->vyaw * dt;
        }

        // Interpolate between prev and it
        auto prev = std::prev(it);
        double dt = static_cast<double>(
            static_cast<int64_t>(target_ns - prev->timestamp_ns)) * 1e-9;
        return prev->yaw + prev->vyaw * dt;
    }

    // ---------------------------------------------------------------------------
    // Whacknet-style constrained PnP
    //
    // Rotation is known from the gyro yaw; only 3-DOF translation is solved via
    // an 8×3 overdetermined linear system (one 2-row block per tag corner).
    //
    // Math: for each corner i with Q_i = R_cam_tag * P_tag_i and image point (u,v):
    //   [fx,  0, cx−u] [tx]   [(u−cx)·Q_z − fx·Q_x]
    //   [ 0, fy, cy−v] [ty] = [(v−cy)·Q_z − fy·Q_y]
    //                  [tz]
    // Solved via normal equations + SVD (AtA is 3×3 → stable).
    // ---------------------------------------------------------------------------
    std::optional<VisionPoseResult> constrained_pnp(
        const TagPose& tp,
        double yaw_rad,
        const std::vector<cv::Point2d>& img_pts,
        const std::vector<cv::Point3d>& obj_pts,
        uint64_t capture_ns)
    {
        // Undistort image points to corrected pixel coords (same K, distortion removed)
        std::vector<cv::Point2d> pts_ud;
        cv::undistortPoints(img_pts, pts_ud, cam_mat, dist, cv::noArray(), cam_mat);

        // R_field_cam = Rz(gyro_yaw) * R_robot_cam
        cv::Mat Rz_yaw      = make_transform(0, 0, 0, 0, 0, yaw_rad)(cv::Rect(0, 0, 3, 3));
        cv::Mat R_robot_cam = T_robot_cam(cv::Rect(0, 0, 3, 3));
        cv::Mat R_field_cam = Rz_yaw * R_robot_cam;

        // R_cam_tag = R_field_cam^T * R_field_tag
        cv::Mat T_field_tag = make_transform(tp.x, tp.y, tp.z, tp.roll, tp.pitch, tp.yaw);
        cv::Mat R_field_tag = T_field_tag(cv::Rect(0, 0, 3, 3));
        cv::Mat R_cam_tag   = R_field_cam.t() * R_field_tag;

        const double fx = cam_mat.at<double>(0, 0), fy = cam_mat.at<double>(1, 1);
        const double cx = cam_mat.at<double>(0, 2), cy = cam_mat.at<double>(1, 2);

        cv::Mat A(8, 3, CV_64F), b_vec(8, 1, CV_64F);
        for (int i = 0; i < 4; ++i) {
            cv::Mat P = (cv::Mat_<double>(3, 1) << obj_pts[i].x, obj_pts[i].y, obj_pts[i].z);
            cv::Mat Q = R_cam_tag * P;
            const double Qx = Q.at<double>(0), Qy = Q.at<double>(1), Qz = Q.at<double>(2);
            const double u = pts_ud[i].x, v = pts_ud[i].y;

            A.at<double>(2*i,   0) = fx;  A.at<double>(2*i,   1) = 0;   A.at<double>(2*i,   2) = cx - u;
            b_vec.at<double>(2*i)  = (u - cx) * Qz - fx * Qx;

            A.at<double>(2*i+1, 0) = 0;   A.at<double>(2*i+1, 1) = fy;  A.at<double>(2*i+1, 2) = cy - v;
            b_vec.at<double>(2*i+1) = (v - cy) * Qz - fy * Qy;
        }

        cv::Mat AtA = A.t() * A;   // 3×3 — numerically stable for SVD
        cv::Mat Atb = A.t() * b_vec;
        cv::Mat t_solved;
        if (!cv::solve(AtA, Atb, t_solved, cv::DECOMP_SVD)) return std::nullopt;
        if (t_solved.at<double>(2) <= 0.05) return std::nullopt; // tag must be in front of camera

        // Assemble T_cam_tag and propagate to field-robot
        cv::Mat T_cam_tag = cv::Mat::eye(4, 4, CV_64F);
        R_cam_tag.copyTo(T_cam_tag(cv::Rect(0, 0, 3, 3)));
        t_solved.copyTo(T_cam_tag(cv::Rect(3, 0, 1, 3)));

        cv::Mat T_field_robot = T_field_tag * invert_transform(T_cam_tag) * invert_transform(T_robot_cam);
        const double rx = T_field_robot.at<double>(0, 3);
        const double ry = T_field_robot.at<double>(1, 3);

        // Use gyro yaw directly — it's what constructed R_cam_tag, so re-extracting from T is redundant
        return VisionPoseResult{rx, ry, yaw_rad, capture_ns};
    }

    // ---------------------------------------------------------------------------
    // IPPE_SQUARE fallback with floor-constraint disambiguation
    //
    // When no gyro stream is available, IPPE gives two solutions.
    // The correct one has robot z ≈ 0 (robot drives on the floor).
    // If ambiguity ratio is very high (> 0.25), the detection is too uncertain to use.
    // ---------------------------------------------------------------------------
    std::optional<VisionPoseResult> ippe_pnp(
        const TagPose& tp,
        const std::vector<cv::Point2d>& img_pts,
        const std::vector<cv::Point3d>& obj_pts,
        uint64_t capture_ns)
    {
        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<double>  errors;
        int n = cv::solvePnPGeneric(obj_pts, img_pts, cam_mat, dist,
                                    rvecs, tvecs, false,
                                    cv::SOLVEPNP_IPPE_SQUARE,
                                    cv::noArray(), cv::noArray(), errors);
        if (n < 1) return std::nullopt;

        double ambiguity = (n >= 2 && errors[1] > 1e-6) ? (errors[0] / errors[1]) : 0.0;
        if (ambiguity > 0.25) return std::nullopt;

        cv::Mat T_field_tag = make_transform(tp.x, tp.y, tp.z, tp.roll, tp.pitch, tp.yaw);

        auto to_field_robot = [&](int s) -> cv::Mat {
            cv::Mat R; cv::Rodrigues(rvecs[s], R);
            cv::Mat T_cam_tag = cv::Mat::eye(4, 4, CV_64F);
            R.copyTo(T_cam_tag(cv::Rect(0, 0, 3, 3)));
            tvecs[s].copyTo(T_cam_tag(cv::Rect(3, 0, 1, 3)));
            return T_field_tag * invert_transform(T_cam_tag) * invert_transform(T_robot_cam);
        };

        // Pick the solution whose robot origin is closer to z=0 (on the floor)
        int sol = 0;
        if (n >= 2 && ambiguity > 0.15) {
            cv::Mat T0 = to_field_robot(0), T1 = to_field_robot(1);
            if (std::abs(T1.at<double>(2, 3)) < std::abs(T0.at<double>(2, 3))) sol = 1;
        }

        cv::Mat T_field_robot = to_field_robot(sol);
        double rx  = T_field_robot.at<double>(0, 3);
        double ry  = T_field_robot.at<double>(1, 3);
        double yaw = std::atan2(T_field_robot.at<double>(1, 0),
                                T_field_robot.at<double>(0, 0));

        return VisionPoseResult{rx, ry, yaw, capture_ns};
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

AprilTagDetector::AprilTagDetector(AprilTagLayout layout)
    : impl_(new Impl(std::move(layout))) {}

AprilTagDetector::~AprilTagDetector() { delete impl_; }

bool AprilTagDetector::is_open() const {
    return impl_->v4l2_fd >= 0 && impl_->det != nullptr;
}

void AprilTagDetector::update_gyro(double yaw, double vyaw, uint64_t timestamp_ns) {
    std::lock_guard<std::mutex> lock(impl_->pose_mutex);
    impl_->pose_history.push_back({timestamp_ns, yaw, vyaw});
    if (impl_->pose_history.size() > POSE_HISTORY_SIZE)
        impl_->pose_history.pop_front();
}

std::optional<VisionPoseResult> AprilTagDetector::detect() {
    if (!is_open()) return std::nullopt;

    // Block until a frame is ready (DQBUF blocks in non-O_NONBLOCK mode)
    struct v4l2_buffer buf = {};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(impl_->v4l2_fd, VIDIOC_DQBUF, &buf) < 0)
        return std::nullopt;

    // Kernel-provided timestamp (monotonic) — same domain as pose_history timestamps
    const uint64_t capture_ns =
        static_cast<uint64_t>(buf.timestamp.tv_sec)  * 1'000'000'000ULL +
        static_cast<uint64_t>(buf.timestamp.tv_usec) * 1'000ULL;

    // YUYV: each pixel pair is [Y0 U Y1 V]; Y bytes at even offsets = grayscale
    const auto* yuyv = static_cast<const uint8_t*>(impl_->buf_start[buf.index]);
    const int   npix = impl_->layout.camera.width * impl_->layout.camera.height;
    for (int i = 0; i < npix; ++i)
        impl_->graybuf[i] = yuyv[i * 2];

    // Return buffer to kernel before heavy compute
    ioctl(impl_->v4l2_fd, VIDIOC_QBUF, &buf);

    image_u8_t img {
        impl_->layout.camera.width,
        impl_->layout.camera.height,
        impl_->layout.camera.width,  // stride = width (contiguous)
        impl_->graybuf.data()
    };

    zarray_t* dets = apriltag_detector_detect(impl_->det, &img);

    const double hs = impl_->layout.tag_size_meters / 2.0;
    // Tag corners in tag-local space (apriltag convention: CCW from bottom-left)
    const std::vector<cv::Point3d> obj_pts = {
        {-hs,  hs, 0},
        { hs,  hs, 0},
        { hs, -hs, 0},
        {-hs, -hs, 0},
    };

    // Interpolate yaw once per frame (shared across all detected tags this frame)
    auto yaw_opt = impl_->interpolate_yaw(capture_ns);

    // Select the highest-confidence detection among known tags
    std::optional<VisionPoseResult> best;
    float best_margin = -1.0f;

    for (int i = 0; i < zarray_size(dets); ++i) {
        apriltag_detection_t* d;
        zarray_get(dets, i, &d);

        auto it = impl_->layout.tags.find(d->id);
        if (it == impl_->layout.tags.end()) continue;

        std::vector<cv::Point2d> img_pts = {
            {d->p[0][0], d->p[0][1]},
            {d->p[1][0], d->p[1][1]},
            {d->p[2][0], d->p[2][1]},
            {d->p[3][0], d->p[3][1]},
        };

        std::optional<VisionPoseResult> result;

        if (yaw_opt.has_value()) {
            result = impl_->constrained_pnp(it->second, *yaw_opt, img_pts, obj_pts, capture_ns);
            if (result) {
                std::fprintf(stderr, "[apriltag] tag %d constrained solve ok (%.2f,%.2f) yaw=%.2f\n",
                             d->id, result->x, result->y, result->heading_rad);
            }
        }

        if (!result) {
            // Constrained solve unavailable or failed — fall back to IPPE with floor constraint
            result = impl_->ippe_pnp(it->second, img_pts, obj_pts, capture_ns);
            if (result) {
                std::fprintf(stderr, "[apriltag] tag %d IPPE fallback (%.2f,%.2f) yaw=%.2f\n",
                             d->id, result->x, result->y, result->heading_rad);
            }
        }

        if (result && d->decision_margin > best_margin) {
            best_margin = d->decision_margin;
            best = result;
        }
    }

    apriltag_detections_destroy(dets);
    return best;
}
