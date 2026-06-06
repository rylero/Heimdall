#include "config/camera_config_loader.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <stdexcept>

namespace fs = std::filesystem;
using json   = nlohmann::json;

static CameraType parse_type(const std::string& s) {
    if (s == "usb") return CameraType::USB;
    if (s == "csi") return CameraType::CSI;
    throw std::runtime_error("unknown camera type '" + s + "' (expected usb or csi)");
}

CameraLoadResult load_camera_configs(const std::string& dir) {
    if (!fs::is_directory(dir))
        throw std::runtime_error("camera config dir not found: " + dir);

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".json")
            paths.push_back(entry.path());
    }
    if (paths.empty())
        throw std::runtime_error("no camera .json files found in: " + dir);

    std::sort(paths.begin(), paths.end());

    CameraLoadResult result;
    for (const auto& p : paths) {
        std::ifstream f(p);
        if (!f) throw std::runtime_error("cannot open " + p.string());
        json j = json::parse(f, nullptr, /*exceptions=*/true, /*ignore_comments=*/true);

        CameraConfig cfg;
        cfg.id        = j.at("id").get<int>();
        cfg.type      = parse_type(j.at("type").get<std::string>());
        cfg.device    = j.value("device",    "");
        cfg.width     = j.value("width",     640);
        cfg.height    = j.value("height",    480);
        cfg.fps       = j.value("fps",       60);
        cfg.hw_decode = j.value("hw_decode", true);
        cfg.mirror_of = j.value("mirror_of", -1);
        cfg.flip_h    = j.value("flip_h", false);
        cfg.flip_v    = j.value("flip_v", false);

        const auto& intr = j.at("intrinsics");
        const auto& extr = j.at("extrinsics");

        CameraParams params;
        params.intrinsics.fx = intr.at("fx").get<float>();
        params.intrinsics.fy = intr.at("fy").get<float>();
        params.intrinsics.cx = intr.at("cx").get<float>();
        params.intrinsics.cy = intr.at("cy").get<float>();
        if (intr.contains("distortion")) {
            const auto& d = intr.at("distortion");
            params.intrinsics.k1 = d.value("k1", 0.f);
            params.intrinsics.k2 = d.value("k2", 0.f);
            params.intrinsics.p1 = d.value("p1", 0.f);
            params.intrinsics.p2 = d.value("p2", 0.f);
            params.intrinsics.k3 = d.value("k3", 0.f);
        }
        params.extrinsics.tx = extr.at("tx").get<float>();
        params.extrinsics.ty = extr.at("ty").get<float>();
        params.extrinsics.tz = extr.at("tz").get<float>();
        params.extrinsics.R  = rotation_from_euler(
            extr.at("yaw").get<float>(),
            extr.at("pitch").get<float>(),
            extr.at("roll").get<float>()
        );

        // Flips happen before nvinfer, so bbox coords are in flipped image space.
        // For an inverted camera (180° roll), the corrected axes become:
        //   std_u = (px_flip - (W-1-cx)) / fx
        //   std_v = (py_flip - (H-1-cy)) / fy
        // fx/fy stay positive — only the principal point is mirrored.
        if (cfg.flip_h)
            params.intrinsics.cx = static_cast<float>(cfg.width)  - 1.f - params.intrinsics.cx;
        if (cfg.flip_v)
            params.intrinsics.cy = static_cast<float>(cfg.height) - 1.f - params.intrinsics.cy;

        result.pipeline_cameras.push_back(cfg);
        result.pose_cameras.push_back(params);
    }
    return result;
}
