#pragma once
#include "pipeline/detection.h"
#include "pose/camera_params.h"
#include "tracker/track_event.h"
#include <optional>
#include <string>
#include <vector>
#include <zmq.hpp>

class CommLayer {
public:
    struct Config {
        std::string pose_bind_addr;        // Jetson PULL — receives robot pose
        std::string output_bind_addr;      // Jetson PUSH — sends track events
        std::string raw_output_bind_addr;  // Jetson PUB  — raw pixel detections (empty = disabled)
        std::string frame_bind_addr;       // Jetson PUB  — JPEG frames for web preview (empty = disabled)
    };

    explicit CommLayer(Config config);

    std::optional<RobotPose> try_recv_pose();

    void send_frame(const std::vector<TrackEvent>& events,
                    uint64_t timestamp_ns,
                    bool healthy = true);

    // Publish raw pixel-space detections for the web UI debug feed.
    // No-op if raw_output_bind_addr was empty at construction.
    void publish_raw(const std::vector<Detection>& detections,
                     uint64_t timestamp_ns);

    void publish_frame(const uint8_t* jpeg_data, size_t size);

    zmq::context_t& context() { return ctx_; }

private:
    zmq::context_t ctx_;
    zmq::socket_t  pull_sock_;
    zmq::socket_t  push_sock_;
    zmq::socket_t  raw_pub_sock_;
    zmq::socket_t  frame_pub_sock_;
    bool           raw_enabled_   = false;
    bool           frame_enabled_ = false;
};
