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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cerrno>
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
    cv::Mat R = T(cv::Rect(0,0,3,3)).t();
    cv::Mat t = T(cv::Rect(3,0,1,3));
    cv::Mat t_inv = -R * t;
    cv::Mat T_inv = cv::Mat::eye(4,4,CV_64F);
    R.copyTo(T_inv(cv::Rect(0,0,3,3)));
    t_inv.copyTo(T_inv(cv::Rect(3,0,1,3)));
    return T_inv;
}

// ---------------------------------------------------------------------------
// Pimpl — V4L2 camera capture (YUYV → grayscale Y-channel, no FFmpeg dep)
// ---------------------------------------------------------------------------

static constexpr int V4L2_NUM_BUFS = 2;

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
};

AprilTagDetector::AprilTagDetector(AprilTagLayout layout)
    : impl_(new Impl(std::move(layout))) {}

AprilTagDetector::~AprilTagDetector() { delete impl_; }

bool AprilTagDetector::is_open() const {
    return impl_->v4l2_fd >= 0 && impl_->det != nullptr;
}

std::optional<VisionPoseResult> AprilTagDetector::detect(double cur_x, double cur_y) {
    if (!is_open()) return std::nullopt;

    // Block until a frame is ready (DQBUF blocks in non-O_NONBLOCK mode)
    struct v4l2_buffer buf = {};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(impl_->v4l2_fd, VIDIOC_DQBUF, &buf) < 0)
        return std::nullopt;

    // Kernel-provided timestamp (monotonic)
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
    std::vector<cv::Point3d> obj_pts = {
        {-hs,  hs, 0},
        { hs,  hs, 0},
        { hs, -hs, 0},
        {-hs, -hs, 0},
    };

    std::optional<VisionPoseResult> best;
    double best_dist2 = std::numeric_limits<double>::max();

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

        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<double>  errors;
        int n = cv::solvePnPGeneric(obj_pts, img_pts,
                                    impl_->cam_mat, impl_->dist,
                                    rvecs, tvecs, false,
                                    cv::SOLVEPNP_IPPE_SQUARE,
                                    cv::noArray(), cv::noArray(), errors);
        if (n < 1) continue;

        double ambiguity = (n >= 2 && errors[1] > 1e-6) ? (errors[0] / errors[1]) : 0.0;

        int sol = 0;
        if (n >= 2 && ambiguity > 0.15) {
            auto eval = [&](int s) -> double {
                const auto& tp = it->second;
                cv::Mat T_field_tag = make_transform(tp.x, tp.y, tp.z,
                                                     tp.roll, tp.pitch, tp.yaw);
                cv::Mat R; cv::Rodrigues(rvecs[s], R);
                cv::Mat T_cam_tag = cv::Mat::eye(4,4,CV_64F);
                R.copyTo(T_cam_tag(cv::Rect(0,0,3,3)));
                tvecs[s].copyTo(T_cam_tag(cv::Rect(3,0,1,3)));
                cv::Mat T_field_cam   = T_field_tag * invert_transform(T_cam_tag);
                cv::Mat T_field_robot = T_field_cam * invert_transform(impl_->T_robot_cam);
                double rx = T_field_robot.at<double>(0,3);
                double ry = T_field_robot.at<double>(1,3);
                return (rx - cur_x)*(rx - cur_x) + (ry - cur_y)*(ry - cur_y);
            };
            if (eval(1) < eval(0)) sol = 1;
        } else if (ambiguity > 0.25) {
            continue;
        }

        const auto& tp = it->second;
        cv::Mat T_field_tag = make_transform(tp.x, tp.y, tp.z,
                                             tp.roll, tp.pitch, tp.yaw);
        cv::Mat R; cv::Rodrigues(rvecs[sol], R);
        cv::Mat T_cam_tag = cv::Mat::eye(4,4,CV_64F);
        R.copyTo(T_cam_tag(cv::Rect(0,0,3,3)));
        tvecs[sol].copyTo(T_cam_tag(cv::Rect(3,0,1,3)));

        cv::Mat T_field_cam   = T_field_tag * invert_transform(T_cam_tag);
        cv::Mat T_field_robot = T_field_cam * invert_transform(impl_->T_robot_cam);

        double rx  = T_field_robot.at<double>(0,3);
        double ry  = T_field_robot.at<double>(1,3);
        double yaw = std::atan2(T_field_robot.at<double>(1,0),
                                T_field_robot.at<double>(0,0));

        double dist2 = (rx - cur_x)*(rx - cur_x) + (ry - cur_y)*(ry - cur_y);
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best = VisionPoseResult{rx, ry, yaw, capture_ns};
        }
    }

    apriltag_detections_destroy(dets);
    return best;
}
