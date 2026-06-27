#include "comm_layer.h"
#include "heimdall.pb.h"
#include <time.h>

CommLayer::CommLayer(Config config)
    : ctx_(1),
      pull_sock_(ctx_, zmq::socket_type::pull),
      pub_sock_(ctx_, zmq::socket_type::pub),
      vision_pub_sock_(ctx_, zmq::socket_type::pub)
{
    pull_sock_.set(zmq::sockopt::rcvtimeo, 0);
    pull_sock_.bind(config.pose_bind_addr);
    pub_sock_.bind(config.output_bind_addr);

    if (!config.vision_pose_bind_addr.empty()) {
        vision_pub_sock_.bind(config.vision_pose_bind_addr);
        vision_enabled_ = true;
    }
}

std::optional<CommLayer::TimestampedPose> CommLayer::try_recv_pose() {
    zmq::message_t msg;
    const auto result = pull_sock_.recv(msg, zmq::recv_flags::dontwait);
    if (!result) return std::nullopt;

    heimdall::RobotPoseMsg proto;
    if (!proto.ParseFromArray(msg.data(), static_cast<int>(msg.size())))
        return std::nullopt;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const uint64_t recv_ns = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
                           + static_cast<uint64_t>(ts.tv_nsec);

    return TimestampedPose{
        RobotPose{proto.x(), proto.y(), proto.heading(), proto.vyaw(), proto.timestamp_ns()},
        recv_ns
    };
}

void CommLayer::send_frame(const std::vector<TrackEvent>& events,
                            uint64_t timestamp_ns,
                            bool healthy) {
    heimdall::DetectionFrameMsg frame;
    frame.set_timestamp_ns(timestamp_ns);
    frame.set_healthy(healthy);

    for (const auto& ev : events) {
        auto* msg_ev = frame.add_events();
        switch (ev.type) {
            case TrackEventType::CONFIRMED: msg_ev->set_type(heimdall::CONFIRMED); break;
            case TrackEventType::UPDATED:   msg_ev->set_type(heimdall::UPDATED);   break;
            case TrackEventType::LOST:      msg_ev->set_type(heimdall::LOST);      break;
            default: break;
        }
        auto* obj = msg_ev->mutable_object();
        obj->set_track_id(ev.object.track_id);
        obj->set_class_id(ev.object.class_id);
        obj->set_x(ev.object.x);
        obj->set_y(ev.object.y);
        obj->set_vx(ev.object.vx);
        obj->set_vy(ev.object.vy);
        obj->set_ax(ev.object.ax);
        obj->set_ay(ev.object.ay);
        obj->set_confidence(ev.object.confidence);
    }

    std::string bytes = frame.SerializeAsString();
    try {
        pub_sock_.send(zmq::buffer(bytes), zmq::send_flags::dontwait);
    } catch (const zmq::error_t&) {
        // No subscriber connected — drop frame rather than stall the pipeline thread.
    }
}

void CommLayer::send_vision_pose(float x, float y, float heading_rad, uint64_t timestamp_ns) {
    if (!vision_enabled_) return;

    heimdall::VisionPoseMsg msg;
    msg.set_x(x);
    msg.set_y(y);
    msg.set_heading(heading_rad);
    msg.set_timestamp_ns(timestamp_ns);

    std::string bytes = msg.SerializeAsString();
    try {
        vision_pub_sock_.send(zmq::buffer(bytes), zmq::send_flags::dontwait);
    } catch (const zmq::error_t&) {}
}


