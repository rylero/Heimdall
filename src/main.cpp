#include "app/heimdall_app.h"
#include "pose/camera_params.h"
#include <csignal>
#include <cstdio>

static HeimdallApp* g_app = nullptr;

static void shutdown(int) {
    if (g_app) g_app->stop();
}

int main() {
    // Camera mounting configuration — adjust for your physical setup.
    // rotation_from_euler(yaw, pitch, roll):
    //   yaw=0 -> camera faces robot forward (+X)
    //   pitch=0.5 -> camera tilted ~28 degrees downward
    std::vector<CameraParams> cameras = {{
        .intrinsics = {500.f, 500.f, 320.f, 240.f},
        .extrinsics = {
            .tx = 0.3f, .ty = 0.f, .tz = 0.6f,
            .R  = rotation_from_euler(0.f, 0.5f, 0.f),
        },
    }};

    HeimdallApp::Config cfg{
        .cameras           = cameras,
        .infer_config_path = "config/infer_rfdetr.txt",
        .tracker           = {
            .confirmation_frames = 3,
            .loss_frames         = 5,
            .gate_distance       = 1.0f,
        },
        .comm = {
            .pose_bind_addr   = "tcp://*:5555",
            .output_bind_addr = "tcp://*:5556",
        },
    };

    HeimdallApp app(cfg);
    g_app = &app;
    std::signal(SIGINT, shutdown);

    std::printf("Heimdall starting. Ctrl+C to stop.\n");
    app.run();
    return 0;
}
