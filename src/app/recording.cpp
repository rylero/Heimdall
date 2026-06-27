#include "app/recording.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

// ── Writer ────────────────────────────────────────────────────────────────────

RecordingWriter::RecordingWriter(const std::string& path)
    : file_(path, std::ios::out | std::ios::trunc)
{
    if (!file_)
        throw std::runtime_error("RecordingWriter: cannot open " + path);
}

void RecordingWriter::write_pose(const CommLayer::TimestampedPose& p) {
    json j = {
        {"t",        "pose"},
        {"recv_ns",  p.jetson_recv_ns},
        {"x",        p.pose.x},
        {"y",        p.pose.y},
        {"heading",  p.pose.heading},
        {"vyaw",     p.pose.vyaw},
        {"ts_ns",    p.pose.timestamp_ns},
    };
    std::lock_guard<std::mutex> lk(mu_);
    file_ << j.dump() << '\n';
}

void RecordingWriter::write_frame(const std::vector<Detection>& dets) {
    json jdets = json::array();
    for (const auto& d : dets) {
        jdets.push_back({
            {"cam",    d.camera_id},
            {"cls",    d.class_id},
            {"conf",   d.confidence},
            {"l",      d.left},
            {"top",    d.top},
            {"w",      d.width},
            {"h",      d.height},
            {"ts_ns",  d.timestamp_ns},
            {"cap_ns", d.capture_monotonic_ns},
        });
    }
    json j = {{"t", "frame"}, {"dets", std::move(jdets)}};
    std::lock_guard<std::mutex> lk(mu_);
    file_ << j.dump() << '\n';
}

// ── Reader ────────────────────────────────────────────────────────────────────

RecordingData load_recording(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_recording: cannot open " + path);

    RecordingData data;
    std::string   line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const json j = json::parse(line);
        const auto t = j.at("t").get<std::string>();

        if (t == "pose") {
            CommLayer::TimestampedPose tp;
            tp.jetson_recv_ns    = j.at("recv_ns").get<uint64_t>();
            tp.pose.x            = j.at("x").get<float>();
            tp.pose.y            = j.at("y").get<float>();
            tp.pose.heading      = j.at("heading").get<float>();
            tp.pose.vyaw         = j.at("vyaw").get<float>();
            tp.pose.timestamp_ns = j.at("ts_ns").get<uint64_t>();
            data.poses.push_back(tp);
        } else if (t == "frame") {
            RecordedFrame frame;
            for (const auto& jd : j.at("dets")) {
                Detection d{};
                d.camera_id            = jd.at("cam").get<int>();
                d.class_id             = jd.at("cls").get<int>();
                d.confidence           = jd.at("conf").get<float>();
                d.left                 = jd.at("l").get<float>();
                d.top                  = jd.at("top").get<float>();
                d.width                = jd.at("w").get<float>();
                d.height               = jd.at("h").get<float>();
                d.timestamp_ns         = jd.at("ts_ns").get<uint64_t>();
                d.capture_monotonic_ns = jd.at("cap_ns").get<uint64_t>();
                frame.dets.push_back(d);
            }
            data.frames.push_back(std::move(frame));
        }
    }
    return data;
}
