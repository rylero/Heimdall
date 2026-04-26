#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <zmq.hpp>
#include "comm/comm_layer.h"
#include "heimdall.pb.h"

using Catch::Matchers::WithinAbs;

// ── Proto roundtrip ────────────────────────────────────────────────────────

TEST_CASE("RobotPoseMsg roundtrip serialization", "[proto]") {
    heimdall::RobotPoseMsg msg;
    msg.set_x(1.5f);
    msg.set_y(-2.3f);
    msg.set_heading(0.785f);
    msg.set_timestamp_ns(123456789ULL);

    std::string bytes = msg.SerializeAsString();
    heimdall::RobotPoseMsg parsed;
    REQUIRE(parsed.ParseFromString(bytes));
    REQUIRE_THAT(parsed.x(),       WithinAbs(1.5f,  1e-5f));
    REQUIRE_THAT(parsed.y(),       WithinAbs(-2.3f, 1e-5f));
    REQUIRE_THAT(parsed.heading(), WithinAbs(0.785f,1e-5f));
    REQUIRE(parsed.timestamp_ns() == 123456789ULL);
}

TEST_CASE("DetectionFrameMsg roundtrip serialization", "[proto]") {
    heimdall::DetectionFrameMsg frame;
    frame.set_timestamp_ns(999ULL);
    frame.set_healthy(true);

    auto* ev = frame.add_events();
    ev->set_type(heimdall::UPDATED);
    auto* obj = ev->mutable_object();
    obj->set_track_id(42);
    obj->set_class_id(1);
    obj->set_x(3.0f);
    obj->set_y(4.0f);
    obj->set_vx(0.5f);
    obj->set_vy(-0.1f);
    obj->set_confidence(0.92f);

    std::string bytes = frame.SerializeAsString();
    heimdall::DetectionFrameMsg parsed;
    REQUIRE(parsed.ParseFromString(bytes));
    REQUIRE(parsed.healthy());
    REQUIRE(parsed.events_size() == 1);
    REQUIRE(parsed.events(0).type()              == heimdall::UPDATED);
    REQUIRE(parsed.events(0).object().track_id() == 42u);
    REQUIRE(parsed.events(0).object().class_id() == 1);
    REQUIRE_THAT(parsed.events(0).object().x(), WithinAbs(3.0f, 1e-5f));
    REQUIRE_THAT(parsed.events(0).object().confidence(), WithinAbs(0.92f, 1e-5f));
}

TEST_CASE("empty DetectionFrameMsg healthy=false", "[proto]") {
    heimdall::DetectionFrameMsg frame;
    frame.set_healthy(false);
    std::string bytes = frame.SerializeAsString();
    heimdall::DetectionFrameMsg parsed;
    REQUIRE(parsed.ParseFromString(bytes));
    REQUIRE_FALSE(parsed.healthy());
    REQUIRE(parsed.events_size() == 0);
}

// ── CommLayer loopback ─────────────────────────────────────────────────────

TEST_CASE("try_recv_pose returns nullopt when no message pending", "[comm]") {
    CommLayer comm({"inproc://pose_empty", "inproc://out_empty"});
    REQUIRE_FALSE(comm.try_recv_pose().has_value());
}

TEST_CASE("try_recv_pose receives and deserializes robot pose", "[comm]") {
    CommLayer comm({"inproc://pose_recv", "inproc://out_recv"});

    // Send a pose directly via raw ZMQ on the same context
    zmq::socket_t sender(comm.context(), zmq::socket_type::push);
    sender.connect("inproc://pose_recv");

    heimdall::RobotPoseMsg pose_msg;
    pose_msg.set_x(7.f);
    pose_msg.set_y(-1.f);
    pose_msg.set_heading(1.57f);
    pose_msg.set_timestamp_ns(500ULL);
    std::string bytes = pose_msg.SerializeAsString();
    sender.send(zmq::buffer(bytes), zmq::send_flags::none);

    auto result = comm.try_recv_pose();
    REQUIRE(result.has_value());
    REQUIRE_THAT(result->x,       WithinAbs(7.f,   1e-4f));
    REQUIRE_THAT(result->y,       WithinAbs(-1.f,  1e-4f));
    REQUIRE_THAT(result->heading, WithinAbs(1.57f, 1e-4f));
}

TEST_CASE("send_frame serializes and delivers detection frame", "[comm]") {
    CommLayer comm({"inproc://pose_send", "inproc://out_send"});

    zmq::socket_t receiver(comm.context(), zmq::socket_type::pull);
    receiver.connect("inproc://out_send");

    TrackedObject obj{1, 3, 2.f, 5.f, 0.1f, -0.2f, 0.88f};
    TrackEvent ev{TrackEventType::CONFIRMED, obj};
    comm.send_frame({ev}, 12345ULL, true);

    zmq::message_t msg;
    REQUIRE(receiver.recv(msg, zmq::recv_flags::none));

    heimdall::DetectionFrameMsg frame;
    REQUIRE(frame.ParseFromArray(msg.data(), static_cast<int>(msg.size())));
    REQUIRE(frame.healthy());
    REQUIRE(frame.timestamp_ns() == 12345ULL);
    REQUIRE(frame.events_size() == 1);
    REQUIRE(frame.events(0).type()              == heimdall::CONFIRMED);
    REQUIRE(frame.events(0).object().track_id() == 1u);
    REQUIRE(frame.events(0).object().class_id() == 3);
    REQUIRE_THAT(frame.events(0).object().x(), WithinAbs(2.f, 1e-4f));
}
