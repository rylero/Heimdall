#include "apriltag_detector.h"
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#ifdef HEIMDALL_APRILTAG_DEBUG_STREAM
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#endif
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <time.h>
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

static double wrap_angle(double a) {
    while (a > M_PI)  a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
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
static constexpr int V4L2_NUM_BUFS = 1;

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

    // Self-calibrated odometry->field yaw offset (radians). Lets the constrained
    // solve work without depending on a precise robot lineup or manual heading
    // zero: it's derived from agreement between low-ambiguity IPPE solves and
    // the fed-in odometry yaw, not assumed correct from t=0.
    double yaw_offset_      = 0.0;
    bool   yaw_calibrated_  = false;
    double calib_candidate_ = 0.0;
    int    calib_streak_    = 0;

    static constexpr double CALIB_AMBIGUITY_MAX   = 0.1;   // stricter than the 0.25 IPPE-fallback cutoff
    static constexpr double CALIB_CONSISTENCY_RAD = 0.05;  // ~3 deg between consecutive candidates
    static constexpr int    CALIB_STREAK_REQUIRED = 3;

    // Called on each confident (low-ambiguity) IPPE solve. Locks/re-locks
    // yaw_offset_ once CALIB_STREAK_REQUIRED consecutive candidates agree,
    // so a single bad disambiguation can't corrupt calibration, and slow
    // odometry drift over a match still gets corrected.
    void update_yaw_calibration(double raw_odometry_yaw, double vision_yaw) {
        double candidate = wrap_angle(vision_yaw - raw_odometry_yaw);
        if (calib_streak_ > 0 &&
            std::abs(wrap_angle(candidate - calib_candidate_)) < CALIB_CONSISTENCY_RAD) {
            ++calib_streak_;
        } else {
            calib_streak_ = 1;
        }
        calib_candidate_ = candidate;
        if (calib_streak_ >= CALIB_STREAK_REQUIRED) {
            yaw_offset_     = candidate;
            yaw_calibrated_ = true;
        }
    }

#ifdef HEIMDALL_APRILTAG_DEBUG_STREAM
    GstElement* debug_pipeline_      = nullptr;
    GstElement* debug_appsrc_        = nullptr;
    uint64_t    debug_pts_ns_        = 0;
    uint64_t    debug_frame_dur_ns_  = 0;
    bool        debug_pipeline_live_ = false;  // deferred until first detect() call
#endif
    bool        debug_ok_         = false;

    Impl(AprilTagLayout lay) : layout(std::move(lay)) {
        auto& c = layout.camera;

        // O_NONBLOCK so DQBUF returns EAGAIN instead of blocking forever if the camera
        // stalls (brownout / USB re-enumeration) — detect() poll()s with a timeout and
        // checks running_ between frames, so stop()/join can't deadlock (5.8).
        v4l2_fd = open(c.device.c_str(), O_RDWR | O_NONBLOCK);
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
        det->quad_decimate = layout.solver.quad_decimate;
        det->quad_sigma    = layout.solver.quad_sigma;
        det->nthreads      = layout.solver.nthreads;
        det->debug         = 0;
        det->refine_edges  = layout.solver.refine_edges ? 1 : 0;

        cam_mat = (cv::Mat_<double>(3,3) <<
            c.fx, 0,    c.cx,
            0,    c.fy, c.cy,
            0,    0,    1);
        dist = (cv::Mat_<double>(1,5) << c.k1, c.k2, c.p1, c.p2, c.k3);

        auto& r = layout.robot_to_camera;
        T_robot_cam = make_transform(r.x, r.y, r.z, r.roll, r.pitch, r.yaw);

#ifdef HEIMDALL_APRILTAG_DEBUG_STREAM
        // Debug RTSP stream via GStreamer appsrc → x264enc → RTMP → MediaMTX.
        gst_init(nullptr, nullptr);
        GError* gst_err = nullptr;
        std::string gst_desc =
            "appsrc name=src format=time is-live=true do-timestamp=false ! "
            "videoconvert ! "
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 ! "
            "flvmux streamable=true ! "
            "rtmpsink location=rtmp://127.0.0.1:1935/live/apriltag";
        debug_pipeline_ = gst_parse_launch(gst_desc.c_str(), &gst_err);
        if (gst_err) {
            std::fprintf(stderr, "[apriltag] debug pipeline error: %s\n", gst_err->message);
            g_clear_error(&gst_err);
        } else {
            debug_appsrc_ = gst_bin_get_by_name(GST_BIN(debug_pipeline_), "src");
            GstCaps* caps = gst_caps_new_simple("video/x-raw",
                "format",    G_TYPE_STRING,       "BGR",
                "width",     G_TYPE_INT,           c.width,
                "height",    G_TYPE_INT,           c.height,
                "framerate", GST_TYPE_FRACTION,    c.fps > 0 ? c.fps : 30, 1,
                nullptr);
            gst_app_src_set_caps(GST_APP_SRC(debug_appsrc_), caps);
            gst_caps_unref(caps);
            g_object_set(debug_appsrc_, "max-bytes", (guint64)0, "block", FALSE, nullptr);
            // Pipeline is built but NOT started here — started lazily on first detect()
            // to avoid NvMap/CMA competition with the DeepStream pipeline at startup.
            debug_frame_dur_ns_ = c.fps > 0 ? (1'000'000'000ULL / (uint64_t)c.fps)
                                             : 33'333'333ULL;
            debug_ok_ = true;
        }
#endif
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
#ifdef HEIMDALL_APRILTAG_DEBUG_STREAM
        if (debug_pipeline_) {
            gst_element_set_state(debug_pipeline_, GST_STATE_NULL);
            gst_object_unref(debug_pipeline_);
        }
        if (debug_appsrc_) gst_object_unref(debug_appsrc_);
#endif
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
            // target is before the whole history. Bound the backward extrapolation: if the
            // capture time predates the oldest sample by more than the staleness window it's
            // implausible (e.g. a bad capture timestamp), so reject rather than extrapolate
            // yaw over a large negative dt (5.5).
            if (it->timestamp_ns > target_ns &&
                it->timestamp_ns - target_ns > GYRO_STALE_NS)
                return std::nullopt;
            double dt = static_cast<double>(
                static_cast<int64_t>(target_ns - it->timestamp_ns)) * 1e-9;
            return it->yaw + it->vyaw * dt;
        }

        // Linearly interpolate between the two straddling samples prev (< target) and it
        // (>= target), using the true slope between them — not prev's stored vyaw (5.12).
        auto prev = std::prev(it);
        const double span = static_cast<double>(
            static_cast<int64_t>(it->timestamp_ns - prev->timestamp_ns));
        if (span <= 0.0) return prev->yaw;
        const double frac = static_cast<double>(
            static_cast<int64_t>(target_ns - prev->timestamp_ns)) / span;
        const double dyaw = wrap_angle(it->yaw - prev->yaw);  // shortest-arc, handles wrap
        return wrap_angle(prev->yaw + frac * dyaw);
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
        uint64_t capture_ns,
        cv::Mat* out_rvec = nullptr,
        cv::Mat* out_tvec = nullptr)
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

        if (out_rvec) cv::Rodrigues(R_cam_tag, *out_rvec);
        if (out_tvec) *out_tvec = t_solved.clone();

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
        uint64_t capture_ns,
        cv::Mat* out_rvec = nullptr,
        cv::Mat* out_tvec = nullptr,
        double*  out_ambiguity = nullptr)
    {
        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<double>  errors;
        int n = cv::solvePnPGeneric(obj_pts, img_pts, cam_mat, dist,
                                    rvecs, tvecs, false,
                                    cv::SOLVEPNP_IPPE_SQUARE,
                                    cv::noArray(), cv::noArray(), errors);
        if (n < 1) return std::nullopt;

        // ambiguity = errors[0]/errors[1] ∈ [0,1]; nearer 1 = the second solution reprojects
        // almost as well (more ambiguous). When errors[1] ≈ 0 BOTH solutions reproject
        // perfectly — the *maximally* ambiguous case — so treat it as ~1 and let the
        // ambiguity_max reject fire, instead of the old code's 0.0 which blindly trusted
        // solution 0 (5.11).
        double ambiguity;
        if (n < 2)                      ambiguity = 0.0;   // single solution — unambiguous
        else if (errors[1] > 1e-6)      ambiguity = errors[0] / errors[1];
        else                            ambiguity = 1.0;   // both near-perfect → maximally ambiguous
        if (out_ambiguity) *out_ambiguity = ambiguity;
        if (ambiguity > layout.solver.ambiguity_max) return std::nullopt;

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
        if (n >= 2 && ambiguity > layout.solver.floor_disambiguation_min) {
            cv::Mat T0 = to_field_robot(0), T1 = to_field_robot(1);
            if (std::abs(T1.at<double>(2, 3)) < std::abs(T0.at<double>(2, 3))) sol = 1;
        }

        cv::Mat T_field_robot = to_field_robot(sol);
        double rx  = T_field_robot.at<double>(0, 3);
        double ry  = T_field_robot.at<double>(1, 3);
        double yaw = std::atan2(T_field_robot.at<double>(1, 0),
                                T_field_robot.at<double>(0, 0));

        if (out_rvec) *out_rvec = rvecs[sol];
        if (out_tvec) *out_tvec = tvecs[sol];

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

    // Wait for a frame with a timeout so a stalled camera can't block shutdown (5.8).
    // The fd is O_NONBLOCK, so poll() gates DQBUF; on timeout we return and the caller
    // re-checks running_.
    struct pollfd pfd { impl_->v4l2_fd, POLLIN, 0 };
    const int pr = poll(&pfd, 1, 200 /* ms */);
    if (pr <= 0 || !(pfd.revents & POLLIN))
        return std::nullopt;  // timeout or error — let the loop re-check running_

    struct v4l2_buffer buf = {};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(impl_->v4l2_fd, VIDIOC_DQBUF, &buf) < 0)
        return std::nullopt;

    // Kernel-provided capture timestamp. It should be CLOCK_MONOTONIC (same domain as the
    // gyro pose_history), but not all UVC drivers set that — some emit 0 or a different
    // clock. If the monotonic flag is missing or the value is 0, fall back to reading
    // CLOCK_MONOTONIC now (5.5). A bogus (e.g. 0) capture time would otherwise send
    // interpolate_yaw back-extrapolating over a multi-second negative dt → garbage yaw.
    uint64_t capture_ns =
        static_cast<uint64_t>(buf.timestamp.tv_sec)  * 1'000'000'000ULL +
        static_cast<uint64_t>(buf.timestamp.tv_usec) * 1'000ULL;
    const bool ts_is_monotonic =
        (buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
    if (!ts_is_monotonic || capture_ns == 0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        capture_ns = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
                   + static_cast<uint64_t>(ts.tv_nsec);
    }

    // YUYV: each pixel pair is [Y0 U Y1 V]; Y bytes at even offsets = grayscale
    const auto* yuyv = static_cast<const uint8_t*>(impl_->buf_start[buf.index]);
    const int   npix = impl_->layout.camera.width * impl_->layout.camera.height;
    for (int i = 0; i < npix; ++i)
        impl_->graybuf[i] = yuyv[i * 2];

    // Build debug frame from grayscale before returning the V4L2 buffer
    cv::Mat debug_frame;
#ifdef HEIMDALL_APRILTAG_DEBUG_STREAM
    if (impl_->debug_ok_) {
        cv::Mat gray(impl_->layout.camera.height, impl_->layout.camera.width,
                     CV_8UC1, impl_->graybuf.data());
        cv::cvtColor(gray, debug_frame, cv::COLOR_GRAY2BGR);
    }
#endif

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
    // Tag corners in tag-local space, paired index-for-index with apriltag's d->p[].
    // apriltag maps tag-normalized coords (-1,-1),(1,-1),(1,1),(-1,1) → p[0..3], which in a
    // tag frame with +x right / +y up is bottom-left, bottom-right, top-right, top-left.
    // The order below is TL,TR,BR,BL — the vertical mirror of that. It is consistent with the
    // solve only if this camera's image y-axis convention flips it back; the yaw
    // self-calibration converging in practice is evidence it lines up, but this has NOT been
    // verified against a known tag at a measured pose. 5.6: verify with a static known-tag
    // frame (cross-check PhotonVision/wpical ordering) before trusting absolute heading; a
    // mismatch yields a low-reprojection but rotated/mirrored pose that looks "locked".
    const std::vector<cv::Point3d> obj_pts = {
        {-hs,  hs, 0},
        { hs,  hs, 0},
        { hs, -hs, 0},
        {-hs, -hs, 0},
    };

    // Interpolate yaw once per frame (shared across all detected tags this frame)
    auto yaw_opt = impl_->interpolate_yaw(capture_ns);

    // Per-tag debug info collected for the overlay
    struct TagDraw {
        int id;
        float margin;
        cv::Point2f corners[4];
        cv::Point2f center;
        std::string method;  // "constrained" | "IPPE" | "failed" | "unknown"
        bool        has_pose;
        cv::Mat     rvec, tvec;
    };
    std::vector<TagDraw> tag_draws;

    // Select the highest-confidence detection among known tags
    std::optional<VisionPoseResult> best;
    float best_margin = -1.0f;

    for (int i = 0; i < zarray_size(dets); ++i) {
        apriltag_detection_t* d;
        zarray_get(dets, i, &d);

        TagDraw td;
        td.id     = d->id;
        td.margin = d->decision_margin;
        for (int j = 0; j < 4; ++j)
            td.corners[j] = {(float)d->p[j][0], (float)d->p[j][1]};
        td.center  = {(float)d->c[0], (float)d->c[1]};
        td.method  = "unknown";
        td.has_pose = false;

        auto it = impl_->layout.tags.find(d->id);
        if (it == impl_->layout.tags.end()) {
            tag_draws.push_back(std::move(td));
            continue;
        }

        std::vector<cv::Point2d> img_pts = {
            {d->p[0][0], d->p[0][1]},
            {d->p[1][0], d->p[1][1]},
            {d->p[2][0], d->p[2][1]},
            {d->p[3][0], d->p[3][1]},
        };

        std::optional<VisionPoseResult> result;
        const bool try_constrained = !impl_->layout.force_unconstrained_solver
                                   && impl_->yaw_calibrated_
                                   && yaw_opt.has_value();

        if (try_constrained) {
            double corrected_yaw = wrap_angle(*yaw_opt + impl_->yaw_offset_);
            result = impl_->constrained_pnp(it->second, corrected_yaw, img_pts, obj_pts, capture_ns,
                                            &td.rvec, &td.tvec);
            if (result) {
                td.method   = "constrained";
                td.has_pose = true;
                std::fprintf(stderr, "[apriltag] tag %d constrained solve ok (%.2f,%.2f) yaw=%.2f\n",
                             d->id, result->x, result->y, result->heading_rad);
            }
        }

        if (!result) {
            double ambiguity = 0.0;
            result = impl_->ippe_pnp(it->second, img_pts, obj_pts, capture_ns,
                                     &td.rvec, &td.tvec, &ambiguity);
            if (result) {
                td.method   = impl_->layout.force_unconstrained_solver ? "IPPE(forced)" : "IPPE";
                td.has_pose = true;
                std::fprintf(stderr, "[apriltag] tag %d IPPE (%.2f,%.2f) yaw=%.2f amb=%.3f\n",
                             d->id, result->x, result->y, result->heading_rad, ambiguity);

                // Feed a confident solve into yaw self-calibration. Skipped in
                // forced-unconstrained mode — there's no constrained solve to calibrate for.
                if (!impl_->layout.force_unconstrained_solver
                    && yaw_opt.has_value()
                    && ambiguity < Impl::CALIB_AMBIGUITY_MAX) {
                    impl_->update_yaw_calibration(*yaw_opt, result->heading_rad);
                }
            } else {
                td.method = "failed";
            }
        }

        if (result && d->decision_margin > best_margin) {
            best_margin = d->decision_margin;
            best = result;
        }
        tag_draws.push_back(std::move(td));
    }

    apriltag_detections_destroy(dets);

    // ── Debug overlay ────────────────────────────────────────────────────────
#ifdef HEIMDALL_APRILTAG_DEBUG_STREAM
    if (impl_->debug_ok_ && !debug_frame.empty()) {
        const cv::Scalar COL_GREEN  {  0, 255,   0};
        const cv::Scalar COL_YELLOW {  0, 255, 255};
        const cv::Scalar COL_CYAN   {255, 255,   0};
        const cv::Scalar COL_RED    {  0,   0, 255};
        const cv::Scalar COL_WHITE  {255, 255, 255};
        const cv::Scalar COL_ORANGE {  0, 165, 255};
        const cv::Scalar COL_AXIS_X {  0,   0, 255};  // red
        const cv::Scalar COL_AXIS_Y {  0, 255,   0};  // green
        const cv::Scalar COL_AXIS_Z {255,   0,   0};  // blue

        const std::vector<cv::Point3d> axis_3d = {
            {0, 0, 0}, {hs, 0, 0}, {0, hs, 0}, {0, 0, hs}};

        for (const auto& td : tag_draws) {
            // Polygon around tag corners
            const cv::Scalar poly_col = td.has_pose ? COL_GREEN : COL_ORANGE;
            for (int j = 0; j < 4; ++j) {
                cv::line(debug_frame,
                         cv::Point(td.corners[j]),
                         cv::Point(td.corners[(j + 1) % 4]),
                         poly_col, 2);
                // Number each corner
                cv::putText(debug_frame, std::to_string(j),
                            cv::Point(td.corners[j]) + cv::Point(4, -4),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, COL_YELLOW, 1);
            }

            // Center dot
            cv::circle(debug_frame, cv::Point(td.center), 5, poly_col, -1);

            // Tag ID + margin
            std::string id_txt = "ID:" + std::to_string(td.id) +
                                 " m=" + std::to_string((int)td.margin);
            cv::putText(debug_frame, id_txt,
                        cv::Point(td.center) + cv::Point(-40, -14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, COL_CYAN, 1);

            // Solve method
            cv::putText(debug_frame, td.method,
                        cv::Point(td.center) + cv::Point(-40, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1);

            // 3-axis transform arrow if we have rvec/tvec
            if (td.has_pose && !td.rvec.empty() && !td.tvec.empty()) {
                std::vector<cv::Point2d> axis_img;
                cv::projectPoints(axis_3d, td.rvec, td.tvec,
                                  impl_->cam_mat, impl_->dist, axis_img);
                cv::Point org(axis_img[0]);
                cv::arrowedLine(debug_frame, org, cv::Point(axis_img[1]), COL_AXIS_X, 2, 8, 0, 0.2);
                cv::arrowedLine(debug_frame, org, cv::Point(axis_img[2]), COL_AXIS_Y, 2, 8, 0, 0.2);
                cv::arrowedLine(debug_frame, org, cv::Point(axis_img[3]), COL_AXIS_Z, 2, 8, 0, 0.2);
                cv::putText(debug_frame, "X", cv::Point(axis_img[1]) + cv::Point(3,-3),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, COL_AXIS_X, 1);
                cv::putText(debug_frame, "Y", cv::Point(axis_img[2]) + cv::Point(3,-3),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, COL_AXIS_Y, 1);
                cv::putText(debug_frame, "Z", cv::Point(axis_img[3]) + cv::Point(3,-3),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, COL_AXIS_Z, 1);
            }
        }

        // Header bar: pose result + gyro status
        char hdr[128];
        if (best) {
            std::snprintf(hdr, sizeof(hdr), "POSE x=%.2f y=%.2f hdg=%.1fdeg  tags=%d",
                          best->x, best->y,
                          best->heading_rad * 180.0 / M_PI,
                          (int)tag_draws.size());
        } else {
            std::snprintf(hdr, sizeof(hdr), "NO POSE  tags=%d",
                          (int)tag_draws.size());
        }
        cv::putText(debug_frame, hdr, {10, 28},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    best ? COL_GREEN : COL_RED, 2);

        const char* gyro_str = yaw_opt.has_value() ? "GYRO:FRESH" : "GYRO:STALE";
        cv::putText(debug_frame, gyro_str, {10, 54},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    yaw_opt.has_value() ? COL_GREEN : COL_ORANGE, 1);

        const char* calib_str = impl_->layout.force_unconstrained_solver ? "YAW:FORCED-IPPE"
                               : impl_->yaw_calibrated_                   ? "YAW:CAL"
                                                                           : "YAW:UNCAL";
        cv::putText(debug_frame, calib_str, {10, 78},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    impl_->yaw_calibrated_ ? COL_GREEN : COL_ORANGE, 1);

#ifdef HEIMDALL_APRILTAG_DEBUG_STREAM
        {
            if (!impl_->debug_pipeline_live_) {
                gst_element_set_state(impl_->debug_pipeline_, GST_STATE_PLAYING);
                impl_->debug_pipeline_live_ = true;
                std::printf("[apriltag] debug stream: rtsp://0.0.0.0:8554/live/apriltag\n");
            }
            const size_t sz = debug_frame.total() * debug_frame.elemSize();
            GstBuffer* gbuf = gst_buffer_new_allocate(nullptr, sz, nullptr);
            GstMapInfo map;
            gst_buffer_map(gbuf, &map, GST_MAP_WRITE);
            std::memcpy(map.data, debug_frame.data, sz);
            gst_buffer_unmap(gbuf, &map);
            GST_BUFFER_PTS(gbuf)      = impl_->debug_pts_ns_;
            GST_BUFFER_DURATION(gbuf) = impl_->debug_frame_dur_ns_;
            impl_->debug_pts_ns_ += impl_->debug_frame_dur_ns_;
            gst_app_src_push_buffer(GST_APP_SRC(impl_->debug_appsrc_), gbuf);
        }
#endif
    }
#endif  // HEIMDALL_APRILTAG_DEBUG_STREAM

    return best;
}
