#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cmath>
#include <string>
#include <thread>
#include <zmq.hpp>
#include "heimdall.pb.h"

static std::atomic<bool> g_running{true};

static void shutdown(int) { g_running = false; }

int main(int argc, char* argv[]) {
    // Optional: pass Jetson IP as first argument (default = localhost)
    const std::string host = (argc > 1) ? argv[1] : "localhost";
    const std::string pose_addr   = "tcp://" + host + ":5555";
    const std::string output_addr = "tcp://" + host + ":5556";

    std::signal(SIGINT, shutdown);

    zmq::context_t ctx(1);

    // PUSH pose to Jetson PULL (port 5555)
    zmq::socket_t push_sock(ctx, zmq::socket_type::push);
    push_sock.connect(pose_addr);

    // PULL track events from Jetson PUSH (port 5556)
    zmq::socket_t pull_sock(ctx, zmq::socket_type::pull);
    pull_sock.set(zmq::sockopt::rcvtimeo, 1);  // 1 ms receive timeout
    pull_sock.connect(output_addr);

    std::printf("Mock robot connected to %s\n", host.c_str());
    std::printf("  Sending pose  -> %s\n", pose_addr.c_str());
    std::printf("  Receiving  <- %s\n\n", output_addr.c_str());

    uint64_t frame = 0;
    auto next_send = std::chrono::steady_clock::now();

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        if (now >= next_send) {
            // Simulate robot driving a slow circle
            const float t = static_cast<float>(frame) * 0.02f;  // seconds at 50 Hz
            heimdall::RobotPoseMsg pose;
            pose.set_x(3.f + 1.f * std::cos(t * 0.2f));
            pose.set_y(3.f + 1.f * std::sin(t * 0.2f));
            pose.set_heading(t * 0.2f);
            pose.set_timestamp_ns(static_cast<uint64_t>(t * 1e9));

            std::string bytes = pose.SerializeAsString();
            push_sock.send(zmq::buffer(bytes), zmq::send_flags::dontwait);

            ++frame;
            next_send += std::chrono::milliseconds(20);  // 50 Hz
        }

        // Non-blocking receive of track events
        zmq::message_t msg;
        if (pull_sock.recv(msg, zmq::recv_flags::dontwait)) {
            heimdall::DetectionFrameMsg frame_msg;
            if (frame_msg.ParseFromArray(msg.data(), static_cast<int>(msg.size()))) {
                std::printf("[ts=%llu healthy=%d events=%d]\n",
                    static_cast<unsigned long long>(frame_msg.timestamp_ns()),
                    static_cast<int>(frame_msg.healthy()),
                    frame_msg.events_size());
                for (int i = 0; i < frame_msg.events_size(); ++i) {
                    const auto& ev  = frame_msg.events(i);
                    const auto& obj = ev.object();
                    const char* type_str =
                        ev.type() == heimdall::CONFIRMED ? "CONFIRMED" :
                        ev.type() == heimdall::UPDATED   ? "UPDATED"   : "LOST";
                    std::printf("  %s id=%u class=%d pos=(%.2f, %.2f) conf=%.2f\n",
                        type_str, obj.track_id(), obj.class_id(),
                        obj.x(), obj.y(), obj.confidence());
                }
            }
        }
    }

    std::printf("\nShutting down mock robot.\n");
    return 0;
}
