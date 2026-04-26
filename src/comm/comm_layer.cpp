#include "comm_layer.h"
#include "heimdall.pb.h"

CommLayer::CommLayer(Config config)
    : ctx_(1),
      pull_sock_(ctx_, zmq::socket_type::pull),
      push_sock_(ctx_, zmq::socket_type::push)
{
    pull_sock_.set(zmq::sockopt::rcvtimeo, 0);  // 0 = non-blocking
    pull_sock_.bind(config.pose_bind_addr);
    push_sock_.bind(config.output_bind_addr);
}

std::optional<RobotPose> CommLayer::try_recv_pose() {
    zmq::message_t msg;
    const auto result = pull_sock_.recv(msg, zmq::recv_flags::dontwait);
    if (!result) return std::nullopt;

    heimdall::RobotPoseMsg proto;
    if (!proto.ParseFromArray(msg.data(), static_cast<int>(msg.size())))
        return std::nullopt;

    return RobotPose{proto.x(), proto.y(), proto.heading()};
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
        }

        auto* obj = msg_ev->mutable_object();
        obj->set_track_id(ev.object.track_id);
        obj->set_class_id(ev.object.class_id);
        obj->set_x(ev.object.x);
        obj->set_y(ev.object.y);
        obj->set_vx(ev.object.vx);
        obj->set_vy(ev.object.vy);
        obj->set_confidence(ev.object.confidence);
    }

    std::string bytes = frame.SerializeAsString();
    push_sock_.send(zmq::buffer(bytes), zmq::send_flags::none);
}
