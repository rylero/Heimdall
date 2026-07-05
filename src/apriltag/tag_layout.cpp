#include "tag_layout.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

AprilTagLayout load_apriltag_layout(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("apriltag layout not found: " + path);

    // We are using ignore_comments here since we use .jsonc files for easier configuration
    nlohmann::json j = nlohmann::json::parse(f, nullptr, /*exceptions=*/true, /*ignore_comments=*/true);

    AprilTagLayout layout;
    layout.tag_size_meters         = j.value("tag_size_meters", 0.1651);
    layout.force_unconstrained_solver = j.value("force_unconstrained_solver", false);

    // Load AprilTag Camera Parameters
    auto& cam = layout.camera;
    auto  jc  = j.at("camera");
    cam.device = jc.at("device").get<std::string>();
    cam.width  = jc.value("width",  640);
    cam.height = jc.value("height", 480);
    cam.fps    = jc.value("fps",    10);
    cam.fx = jc.at("fx").get<double>();
    cam.fy = jc.at("fy").get<double>();
    cam.cx = jc.at("cx").get<double>();
    cam.cy = jc.at("cy").get<double>();
    cam.k1 = jc.value("k1", 0.0);
    cam.k2 = jc.value("k2", 0.0);
    cam.p1 = jc.value("p1", 0.0);
    cam.p2 = jc.value("p2", 0.0);
    cam.k3 = jc.value("k3", 0.0);

    // Robot -> Camera Transform
    auto  jr = j.at("robot_to_camera");
    auto& r  = layout.robot_to_camera;
    r.x     = jr.value("x",     0.0);
    r.y     = jr.value("y",     0.0);
    r.z     = jr.value("z",     0.0);
    r.roll  = jr.value("roll",  0.0);
    r.pitch = jr.value("pitch", 0.0);
    r.yaw   = jr.value("yaw",   0.0);

    // AprilTag Field Map
    for (auto& jt : j.at("tags")) {
        TagPose tp;
        tp.x     = jt.at("x").get<double>();
        tp.y     = jt.at("y").get<double>();
        tp.z     = jt.value("z",     0.0);
        tp.roll  = jt.value("roll",  0.0);
        tp.pitch = jt.value("pitch", 0.0);
        tp.yaw   = jt.at("yaw").get<double>();
        layout.tags[jt.at("id").get<int>()] = tp;
    }

    return layout;
}
