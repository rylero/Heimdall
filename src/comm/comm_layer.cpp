#include "comm_layer.h"
#include "heimdall.pb.h"
#include <chrono>

CommLayer::CommLayer(Config config)
    : ctx_(1),
      pose_pull_sock_(ctx_, zmq::socket_type::pull),
      output_pub_sock_(ctx_, zmq::socket_type::pub),
      apriltag_pose_pub_sock_(ctx_, zmq::socket_type::pub)
{
    // Latest-value semantics (5.4). Without CONFLATE a briefly-slow subscriber accumulates
    // up to SNDHWM (default 1000) queued frames and then reads stale backlog instead of the
    // newest — defeating the "drop stale, never block" design. CONFLATE keeps only the most
    // recent message. Must be set BEFORE bind/connect. (CONFLATE is single-part only, which
    // is fine — these sockets never send multipart.)
    output_pub_sock_.set(zmq::sockopt::conflate, 1);
    pose_pull_sock_.set(zmq::sockopt::rcvtimeo, 0);
    // Keep only the freshest few poses queued; recv loop drains them each cycle.
    pose_pull_sock_.set(zmq::sockopt::rcvhwm, 4);
    pose_pull_sock_.bind(config.pose_bind_addr);
    output_pub_sock_.bind(config.output_bind_addr);

    if (!config.apriltag_pose_bind_addr.empty()) {
        apriltag_pose_pub_sock_.set(zmq::sockopt::conflate, 1);
        apriltag_pose_pub_sock_.bind(config.apriltag_pose_bind_addr);
        apriltag_pose_enabled_ = true;
    }
}

std::optional<CommLayer::TimestampedPose> CommLayer::try_recv_pose() {
    zmq::message_t msg;
    const auto result = pose_pull_sock_.recv(msg, zmq::recv_flags::dontwait);
    if (!result) return std::nullopt;

    heimdall::RobotPoseMsg proto;
    if (!proto.ParseFromArray(msg.data(), static_cast<int>(msg.size())))
        return std::nullopt;

    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const uint64_t recv_ns =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());

    return TimestampedPose{
        RobotPose{proto.x(), proto.y(), proto.heading(), proto.vyaw(), proto.timestamp_ns()},
        recv_ns
    };
}

void CommLayer::send_tracking_frame(const std::vector<TrackEvent>& events,
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
        output_pub_sock_.send(zmq::buffer(bytes), zmq::send_flags::dontwait);
    } catch (const zmq::error_t&) {
        // No subscriber connected — drop frame rather than stall the pipeline thread.
    }
}

void CommLayer::send_apriltag_pose(float x, float y, float heading_rad, uint64_t timestamp_ns) {
    if (!apriltag_pose_enabled_) return;

    heimdall::VisionPoseMsg msg;
    msg.set_x(x);
    msg.set_y(y);
    msg.set_heading(heading_rad);
    msg.set_timestamp_ns(timestamp_ns);

    std::string bytes = msg.SerializeAsString();
    try {
        apriltag_pose_pub_sock_.send(zmq::buffer(bytes), zmq::send_flags::dontwait);
    } catch (const zmq::error_t&) {}
}


