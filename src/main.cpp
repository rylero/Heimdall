#include "pipeline/camera_source.h"
#include "pipeline/detection.h"
#include "pipeline/pipeline.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <vector>

static DeepStreamPipeline* g_pipeline = nullptr;
static std::atomic<int>    g_frames{0};
static auto                g_start = std::chrono::steady_clock::now();

static void on_detection(const std::vector<Detection>& dets) {
    ++g_frames;
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g_start).count();

    for (const auto& d : dets)
        std::printf("cam=%d class=%d conf=%.2f box=(%.0f,%.0f,%.0fx%.0f)\n",
            d.camera_id, d.class_id, d.confidence,
            d.left, d.top, d.width, d.height);

    std::printf("  FPS: %.1f  frames=%d\n",
        static_cast<double>(g_frames.load()) / elapsed,
        g_frames.load());
}

static void shutdown(int) {
    if (g_pipeline) g_pipeline->stop();
}

int main() {
    std::vector<CameraConfig> cameras{
        {0, CameraType::USB, "/dev/video0"},
    };

    DeepStreamPipeline pipeline(cameras, "config/infer_rfdetr.txt", on_detection);
    g_pipeline = &pipeline;
    std::signal(SIGINT, shutdown);

    g_start = std::chrono::steady_clock::now();
    pipeline.run();
    return 0;
}
