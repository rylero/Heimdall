#pragma once
#include "pose/camera_params.h"
#include "tracker/track_event.h"
#include <optional>
#include <string>
#include <vector>
#include <zmq.hpp>

class CommLayer {
public:
    struct Config {
        std::string pose_bind_addr        = "tcp://*:5555";  // Jetson PULL — receives robot pose
        std::string output_bind_addr      = "tcp://*:5556";  // Jetson PUB  — sends track events
        std::string apriltag_pose_bind_addr = "tcp://*:5558";  // Jetson PUB  — AprilTag vision pose
    };

    // Received robot pose tagged with Jetson CLOCK_MONOTONIC reception time.
    struct TimestampedPose {
        RobotPose pose;
        uint64_t  jetson_recv_ns; // CLOCK_MONOTONIC ns when this pose arrived on the Jetson
    };

    explicit CommLayer(Config config);

    std::optional<TimestampedPose> try_recv_pose();

    void send_tracking_frame(const std::vector<TrackEvent>& events,
                             uint64_t timestamp_ns,
                             bool healthy = true);

    // Publish an AprilTag-derived robot pose estimate to port 5558.
    // Robot subscribes and calls addVisionMeasurement().
    // No-op if apriltag_pose_bind_addr was empty at construction.
    void send_apriltag_pose(float x, float y, float heading_rad, uint64_t timestamp_ns);

    zmq::context_t& context() { return ctx_; }

private:
    zmq::context_t ctx_;
    zmq::socket_t  pose_pull_sock_;
    zmq::socket_t  output_pub_sock_;
    zmq::socket_t  apriltag_pose_pub_sock_;
    bool           apriltag_pose_enabled_ = false;
};
