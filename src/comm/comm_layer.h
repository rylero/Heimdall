#pragma once
#include "app/detection_output.h"
#include "pose/camera_params.h"
#include "tracker/track_event.h"
#include <optional>
#include <string>
#include <vector>
#include <zmq.hpp>

class CommLayer : public DetectionOutput {
public:
    struct Config {
        std::string pose_bind_addr        = "tcp://*:5555";  // Jetson PULL — receives robot pose
        std::string output_bind_addr      = "tcp://*:5556";  // Jetson PUB  — sends track events
        std::string apriltag_pose_bind_addr = "tcp://*:5558";  // Jetson PUB  — AprilTag vision pose
    };

    // Received robot pose tagged with a Jetson-domain time estimate for *when the pose was
    // true on the robot* — not the raw reception time. try_recv_pose() maps the robot's own
    // (monotonic) pose timestamp into the Jetson CLOCK_MONOTONIC domain via a min-filtered
    // clock offset, cancelling the per-sample network+scheduling jitter that otherwise
    // corrupts detection↔pose association and the gyro history stamp (§2A.2, 5.7). Kept in the
    // same field so PoseBuffer/replay match unchanged; the value is just jitter-corrected.
    struct TimestampedPose {
        RobotPose pose;
        uint64_t  jetson_recv_ns; // Jetson-domain estimate of the pose's true time (offset-corrected)
    };

    explicit CommLayer(Config config);

    std::optional<TimestampedPose> try_recv_pose();

    void send_tracking_frame(const std::vector<TrackEvent>& events,
                             uint64_t timestamp_ns,
                             bool healthy = true) override;

    // Publish an AprilTag-derived robot pose estimate to port 5558.
    // Robot subscribes and calls addVisionMeasurement(). capture_ns is the frame capture time
    // (CLOCK_MONOTONIC); the message carries latency_ns = now − capture_ns so the robot can
    // place the measurement in its own timebase (§2A.1). The quality fields drive dynamic std
    // devs on the robot (§2B). No-op if apriltag_pose_bind_addr was empty at construction.
    void send_apriltag_pose(float x, float y, float heading_rad, uint64_t capture_ns,
                            uint32_t tag_count, float avg_tag_distance, float reproj_error,
                            float ambiguity, uint32_t solve_mode);

    zmq::context_t& context() { return ctx_; }

private:
    zmq::context_t ctx_;
    zmq::socket_t  pose_pull_sock_;
    zmq::socket_t  output_pub_sock_;
    zmq::socket_t  apriltag_pose_pub_sock_;
    bool           apriltag_pose_enabled_ = false;

    // Clock-offset estimate mapping the robot's pose-timestamp clock into the Jetson
    // CLOCK_MONOTONIC domain (§2A.2). offset ≈ min(jetson_recv − robot_ts): the smallest
    // observed gap corresponds to the least-latency sample, the best offset estimate. Allowed
    // to drift up slowly to track clock skew.
    int64_t        clock_offset_ns_ = 0;
    bool           clock_offset_init_ = false;
};
