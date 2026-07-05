#include "app/snapshot.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace snapshot {

static const char* event_type_str(TrackEventType t) {
    switch (t) {
        case TrackEventType::CONFIRMED: return "confirmed";
        case TrackEventType::UPDATED:   return "updated";
        case TrackEventType::LOST:      return "lost";
    }
    return "unknown";
}

std::string write(const std::string& dir, const Frame& frame) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);  // ok if it already exists

    const uint64_t wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // ----- inputs: AprilTag-derived robot pose -----
    const auto& p = frame.robot_pose;
    const json jpose = {
        {"x", p.x}, {"y", p.y}, {"heading_rad", p.heading},
        {"vyaw", p.vyaw}, {"timestamp_ns", p.timestamp_ns},
    };

    // ----- inputs: raw fuel detections (pixel space) -----
    json jdets = json::array();
    for (const auto& d : frame.detections) {
        jdets.push_back({
            {"camera_id", d.camera_id}, {"class_id", d.class_id},
            {"confidence", d.confidence},
            {"left", d.left}, {"top", d.top}, {"width", d.width}, {"height", d.height},
            {"timestamp_ns", d.timestamp_ns},
            {"capture_monotonic_ns", d.capture_monotonic_ns},
        });
    }

    // ----- intermediate: projected field detections (meters) -----
    json jfield = json::array();
    for (const auto& fd : frame.field_detections) {
        jfield.push_back({
            {"class_id", fd.class_id}, {"x", fd.x}, {"y", fd.y},
            {"confidence", fd.confidence},
        });
    }

    // ----- outputs: tracker events -----
    json jevents = json::array();
    for (const auto& ev : frame.events) {
        jevents.push_back({
            {"type", event_type_str(ev.type)},
            {"track_id", ev.object.track_id}, {"class_id", ev.object.class_id},
            {"x", ev.object.x}, {"y", ev.object.y},
            {"vx", ev.object.vx}, {"vy", ev.object.vy},
            {"ax", ev.object.ax}, {"ay", ev.object.ay},
            {"conf", ev.object.confidence},
        });
    }

    const json j = {
        {"wall_time_ns", wall_ns},
        {"frame_ts_ns", frame.frame_ts_ns},
        {"capture_monotonic_ns", frame.capture_monotonic_ns},
        {"inputs", {
            {"robot_pose", jpose},   // AprilTag raw pose used for projection
            {"detections", jdets},   // fuel raw pixel coords
        }},
        {"intermediate", {
            {"field_detections", jfield},
        }},
        {"outputs", {
            {"healthy", frame.healthy},
            {"events", jevents},
        }},
    };

    const std::string path = dir + "/snapshot_" + std::to_string(frame.frame_ts_ns) + ".json";
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "[snapshot] cannot open %s\n", path.c_str());
        return "";
    }
    const std::string dump = j.dump(2);
    std::fwrite(dump.data(), 1, dump.size(), f);
    std::fputc('\n', f);
    std::fclose(f);
    std::printf("[snapshot] wrote %s\n", path.c_str());
    return path;
}

}  // namespace snapshot
