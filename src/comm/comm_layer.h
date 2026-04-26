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
        std::string pose_bind_addr;    // Jetson PULL — receives robot pose
        std::string output_bind_addr;  // Jetson PUSH — sends track events
    };

    explicit CommLayer(Config config);

    // Non-blocking. Returns the next robot pose if one was waiting, nullopt otherwise.
    std::optional<RobotPose> try_recv_pose();

    // Serialize and send a frame of track events to the robot.
    // healthy=false signals a pipeline error.
    void send_frame(const std::vector<TrackEvent>& events,
                    uint64_t timestamp_ns,
                    bool healthy = true);

    // Expose context so tests can create peer sockets on the same context (for inproc://).
    zmq::context_t& context() { return ctx_; }

private:
    zmq::context_t ctx_;
    zmq::socket_t  pull_sock_;
    zmq::socket_t  push_sock_;
};
