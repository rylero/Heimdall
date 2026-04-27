#include "app/heimdall_app.h"
#include "pipeline/camera_source.h"
#include "pose/camera_params.h"
#include <csignal>
#include <cstdio>

static HeimdallApp* g_app = nullptr;

static void shutdown(int) {
    if (g_app) g_app->stop();
}

int main() {
    // Pipeline cameras: device path + resolution (for GStreamer source elements)
    std::vector<CameraConfig> pipeline_cameras = {
        {0, CameraType::USB, "/dev/video0", 640, 480, 60},
    };

    // Pose cameras: intrinsics + extrinsics (for ground ray projection)
    // rotation_from_euler(yaw, pitch, roll):
    //   yaw=0 -> camera faces robot forward (+X)
    //   pitch=0.5 -> camera tilted ~28 degrees downward
    std::vector<CameraParams> pose_cameras = {{
        .intrinsics = {500.f, 500.f, 320.f, 240.f},
        .extrinsics = {
            .tx = 0.3f, .ty = 0.f, .tz = 0.6f,
            .R  = rotation_from_euler(0.f, 0.5f, 0.f),
        },
    }};

    HeimdallApp::Config cfg{
        .pipeline_cameras  = pipeline_cameras,
        .pose_cameras      = pose_cameras,
        .infer_config_path = "config/infer_rfdetr.txt",
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
