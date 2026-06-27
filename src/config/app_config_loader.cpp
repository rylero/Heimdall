#include "config/app_config_loader.h"
#include "config/camera_config_loader.h"
#include "tracker/track.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

static FilterModel parse_filter_model(const std::string& s) {
    if (s == "constant_position")     return FilterModel::CONSTANT_POSITION;
    if (s == "constant_velocity")     return FilterModel::CONSTANT_VELOCITY;
    if (s == "constant_acceleration") return FilterModel::CONSTANT_ACCELERATION;
    throw std::runtime_error("unknown filter_model '" + s +
        "' (expected constant_position / constant_velocity / constant_acceleration)");
}

HeimdallApp::Config load_app_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open app config: " + path);
    json j = json::parse(f, nullptr, /*exceptions=*/true, /*ignore_comments=*/true);

    std::string cameras_dir = j.value("cameras_dir", "config/cameras");
    auto cameras = load_camera_configs(cameras_dir);

    HeimdallApp::Config cfg;
    cfg.pipeline_cameras  = std::move(cameras.pipeline_cameras);
    cfg.pose_cameras      = std::move(cameras.pose_cameras);
    cfg.infer_config_path = j.value("infer_config", "config/infer_yolo26n.txt");
    cfg.bypass_tracker    = j.value("bypass_tracker", false);
    cfg.log_tracking      = j.value("log_tracking", false);
    cfg.log_path          = j.value("log_path", "logs/tracker_log.csv");

    if (j.contains("apriltag_layout"))
        cfg.apriltag_layout_path = j.at("apriltag_layout").get<std::string>();

    if (j.contains("tracker")) {
        const auto& t = j.at("tracker");
        cfg.tracker.confirmation_frames = t.value("confirmation_frames", 2);
        cfg.tracker.loss_frames         = t.value("loss_frames",         15);
        cfg.tracker.gate_distance       = t.value("gate_distance",       2.0f);
        cfg.tracker.clutter_density     = t.value("clutter_density",     1.0f);
        cfg.tracker.p_detection         = t.value("p_detection",         0.9f);
        if (t.contains("filter_model"))
            cfg.tracker.filter_model = parse_filter_model(
                t.at("filter_model").get<std::string>());
    }

    if (j.contains("comm")) {
        const auto& c = j.at("comm");
        cfg.comm.pose_bind_addr        = c.value("pose_bind_addr",        "tcp://*:5555");
        cfg.comm.output_bind_addr      = c.value("output_bind_addr",      "tcp://*:5556");
        cfg.comm.raw_output_bind_addr  = c.value("raw_output_bind_addr",  "tcp://*:5557");
        cfg.comm.vision_pose_bind_addr = c.value("vision_pose_bind_addr", "tcp://*:5558");
    } else {
        cfg.comm.pose_bind_addr        = "tcp://*:5555";
        cfg.comm.output_bind_addr      = "tcp://*:5556";
        cfg.comm.raw_output_bind_addr  = "tcp://*:5557";
        cfg.comm.vision_pose_bind_addr = "tcp://*:5558";
    }

    return cfg;
}
