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
        // hw_decode=true: Orin Nano has one NvJPEG unit — only one camera can use it
        {.id=0, .type=CameraType::USB, .device="/dev/video0", .width=640, .height=480, .fps=60, .hw_decode=true},
        {.id=1, .type=CameraType::USB, .device="/dev/video2", .width=640, .height=480, .fps=30, .hw_decode=false},
        {.id=2, .type=CameraType::USB, .device="",            .width=640, .height=480, .fps=60, .mirror_of=0},
        {.id=3, .type=CameraType::USB, .device="",            .width=640, .height=480, .fps=30, .mirror_of=1},
    };

    // Pose cameras: intrinsics + extrinsics (for ground ray projection)
    // rotation_from_euler(yaw, pitch, roll):
    //   yaw=0  -> camera faces robot forward (+X)
    //   pitch=0.5 -> camera tilted ~28 degrees downward
    std::vector<CameraParams> pose_cameras = {
        {   // Camera 0: front-left
            .intrinsics = {500.f, 500.f, 320.f, 240.f},
            .extrinsics = {
                .tx = 0.3f, .ty = -0.1f, .tz = 0.6f,
                .R  = rotation_from_euler(0.f, 0.5f, 0.f),
            },
        },
        {   // Camera 1: front-right
            .intrinsics = {500.f, 500.f, 320.f, 240.f},
            .extrinsics = {
                .tx = 0.3f, .ty =  0.1f, .tz = 0.6f,
                .R  = rotation_from_euler(0.f, 0.5f, 0.f),
            },
        },
        {   // Camera 2: test (mirrors cam 0 pose)
            .intrinsics = {500.f, 500.f, 320.f, 240.f},
            .extrinsics = {
                .tx = 0.3f, .ty = -0.1f, .tz = 0.6f,
                .R  = rotation_from_euler(0.f, 0.5f, 0.f),
            },
        },
        {   // Camera 3: test (mirrors cam 1 pose)
            .intrinsics = {500.f, 500.f, 320.f, 240.f},
            .extrinsics = {
                .tx = 0.3f, .ty =  0.1f, .tz = 0.6f,
                .R  = rotation_from_euler(0.f, 0.5f, 0.f),
            },
        },
    };

    HeimdallApp::Config cfg{
        .pipeline_cameras  = pipeline_cameras,
        .pose_cameras      = pose_cameras,
        .infer_config_path = "config/infer_yolo.txt",
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
