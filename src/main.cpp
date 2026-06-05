#include "app/heimdall_app.h"
#include "config/camera_config_loader.h"
#include <csignal>
#include <cstdio>

static HeimdallApp* g_app = nullptr;

static void shutdown(int) {
    if (g_app) g_app->stop();
}

int main() {
    auto cameras = load_camera_configs("config/cameras");

    HeimdallApp::Config cfg{
        .pipeline_cameras  = cameras.pipeline_cameras,
        .pose_cameras      = cameras.pose_cameras,
        .infer_config_path = "config/infer_yolo26n.txt",
        .tracker           = {
            .confirmation_frames = 3,
            .loss_frames         = 5,
            .gate_distance       = 1.0f,
        },
        .comm = {
            .pose_bind_addr       = "tcp://*:5555",
            .output_bind_addr     = "tcp://*:5556",
            .raw_output_bind_addr = "tcp://*:5557",
        },
    };

    HeimdallApp app(cfg);
    g_app = &app;
    std::signal(SIGINT, shutdown);

    std::printf("Heimdall starting. Ctrl+C to stop.\n");
    app.run();
    return 0;
}
