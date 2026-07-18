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
        if (entry.path().extension() == ".jsonc")
            paths.push_back(entry.path());
    }
    if (paths.empty())
        throw std::runtime_error("no camera .jsonc files found in: " + dir);

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
        cfg.rotation  = j.value("rotation", 0);
        cfg.pixel_format = j.value("pixel_format", std::string("image/jpeg"));
        cfg.io_mode      = j.value("io_mode", 2);

        if (cfg.width <= 0 || cfg.height <= 0 || cfg.fps <= 0)
            throw std::runtime_error("camera " + p.string() +
                ": width/height/fps must be positive");
        if (cfg.rotation != 0 && cfg.rotation != 90 &&
            cfg.rotation != 180 && cfg.rotation != 270)
            throw std::runtime_error("camera " + p.string() +
                ": rotation must be one of 0, 90, 180, 270 (got " +
                std::to_string(cfg.rotation) + ")");
        if (j.contains("flip_h") || j.contains("flip_v"))
            throw std::runtime_error("camera " + p.string() +
                ": flip_h/flip_v are no longer supported — use \"rotation\": 0|90|180|270");

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
        // Two ways to specify the camera->robot rotation. Prefer a raw row-major 3x3
        // matrix R (9 floats) for awkward mounts where Euler decomposition order bites;
        // otherwise fall back to yaw/pitch/roll (see rotation_from_euler). The two are
        // mutually exclusive to avoid an ambiguous "which one wins" config.
        const bool has_R     = extr.contains("R");
        const bool has_euler = extr.contains("yaw") || extr.contains("pitch") ||
                               extr.contains("roll");
        if (has_R && has_euler)
            throw std::runtime_error("camera " + p.string() +
                ": extrinsics specify both \"R\" and yaw/pitch/roll — use one or the other");
        if (has_R) {
            const auto& Rj = extr.at("R");
            if (!Rj.is_array() || Rj.size() != 9)
                throw std::runtime_error("camera " + p.string() +
                    ": extrinsics.\"R\" must be an array of 9 numbers (row-major 3x3)");
            for (int i = 0; i < 9; ++i)
                params.extrinsics.R[i] = Rj[i].get<float>();
        } else {
            params.extrinsics.R = rotation_from_euler(
                extr.at("yaw").get<float>(),
                extr.at("pitch").get<float>(),
                extr.at("roll").get<float>()
            );
        }

        // Intrinsics are calibrated on the raw (native) camera feed and are left untouched.
        // The pipeline rotates the frame before nvinfer, so detections arrive rotated; the
        // pose estimator un-rotates each bbox back into this native frame (using rotation +
        // width/height) before unprojecting. This keeps flip-adjustment sign bugs out of the
        // intrinsics entirely — no focal-length negation, no principal-point mirroring.
        params.rotation = cfg.rotation;
        params.width    = cfg.width;
        params.height   = cfg.height;

        result.pipeline_cameras.push_back(cfg);
        result.pose_cameras.push_back(params);
    }
    return result;
}
