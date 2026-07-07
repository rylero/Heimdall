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

    // we use .jsonc for config files that way each setting can have an explainer, thats why comments need to be enabled
    json j = json::parse(f, nullptr, /*exceptions=*/true, /*ignore_comments=*/true);

    std::string cameras_dir = "config/cameras";
    if (j.contains("cameras_dir")) cameras_dir = j.at("cameras_dir").get<std::string>();
    auto cameras = load_camera_configs(cameras_dir);

    HeimdallApp::Config cfg;  // struct defaults are the single source of truth
    cfg.pipeline_cameras = std::move(cameras.pipeline_cameras);
    cfg.pose_cameras     = std::move(cameras.pose_cameras);

    if (j.contains("infer_config"))      cfg.infer_config_path = j.at("infer_config").get<std::string>();
    if (j.contains("bypass_tracker"))    cfg.bypass_tracker    = j.at("bypass_tracker").get<bool>();
    if (j.contains("log_tracking"))      cfg.log_tracking      = j.at("log_tracking").get<bool>();
    if (j.contains("log_path"))          cfg.log_path          = j.at("log_path").get<std::string>();
    if (j.contains("apriltag_layout"))   cfg.apriltag_layout_path = j.at("apriltag_layout").get<std::string>();

    if (j.contains("tracker")) {
        const auto& t = j.at("tracker");
        if (t.contains("confirmation_frames")) cfg.tracker.confirmation_frames = t.at("confirmation_frames").get<int>();
        if (t.contains("loss_frames"))         cfg.tracker.loss_frames         = t.at("loss_frames").get<int>();
        if (t.contains("gate_distance"))       cfg.tracker.gate_distance       = t.at("gate_distance").get<float>();
        if (t.contains("clutter_density"))     cfg.tracker.clutter_density     = t.at("clutter_density").get<float>();
        if (t.contains("p_detection"))         cfg.tracker.p_detection         = t.at("p_detection").get<float>();
        if (t.contains("meas_noise_r"))        cfg.tracker.meas_noise_r        = t.at("meas_noise_r").get<float>();
        if (t.contains("process_noise_q"))     cfg.tracker.process_noise_q     = t.at("process_noise_q").get<float>();
        if (t.contains("pos_cov_floor"))       cfg.tracker.pos_cov_floor       = t.at("pos_cov_floor").get<float>();
        if (t.contains("mahalanobis_gate"))    cfg.tracker.mahalanobis_gate    = t.at("mahalanobis_gate").get<float>();
        if (t.contains("filter_model"))        cfg.tracker.filter_model        = parse_filter_model(t.at("filter_model").get<std::string>());

        // Range validation — catch config typos that would otherwise load silently and break
        // the filter (e.g. the historical meas_noise_r: 0 → division blow-up).
        const auto& tk = cfg.tracker;
        if (tk.meas_noise_r <= 0.f)
            throw std::runtime_error("tracker.meas_noise_r must be > 0");
        if (tk.process_noise_q < 0.f)
            throw std::runtime_error("tracker.process_noise_q must be >= 0");
        if (tk.pos_cov_floor < 0.f)
            throw std::runtime_error("tracker.pos_cov_floor must be >= 0");
        if (tk.gate_distance <= 0.f)
            throw std::runtime_error("tracker.gate_distance must be > 0");
        if (tk.mahalanobis_gate <= 0.f)
            throw std::runtime_error("tracker.mahalanobis_gate must be > 0");
        if (tk.p_detection <= 0.f || tk.p_detection > 1.f)
            throw std::runtime_error("tracker.p_detection must be in (0, 1]");
        if (tk.clutter_density < 0.f)
            throw std::runtime_error("tracker.clutter_density must be >= 0");
        if (tk.confirmation_frames < 1)
            throw std::runtime_error("tracker.confirmation_frames must be >= 1");
        if (tk.loss_frames < 1)
            throw std::runtime_error("tracker.loss_frames must be >= 1");
    }

    if (j.contains("comm")) {
        const auto& c = j.at("comm");
        if (c.contains("pose_bind_addr"))        cfg.comm.pose_bind_addr        = c.at("pose_bind_addr").get<std::string>();
        if (c.contains("output_bind_addr"))      cfg.comm.output_bind_addr      = c.at("output_bind_addr").get<std::string>();
        if (c.contains("apriltag_pose_bind_addr")) cfg.comm.apriltag_pose_bind_addr = c.at("apriltag_pose_bind_addr").get<std::string>();
    }

    return cfg;
}
