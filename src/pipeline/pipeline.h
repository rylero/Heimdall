#pragma once
#include "camera_source.h"
#include "probe.h"
#include <gst/gst.h>
#include <string>
#include <vector>

class DeepStreamPipeline {
public:
    DeepStreamPipeline(
        std::vector<CameraConfig> cameras,
        std::string               infer_config_path,
        DetectionCallback         on_detection
    );
    ~DeepStreamPipeline();

    void build();
    void run();
    void stop();

    // Best-effort recovery from a silent stall (5.16): cycle the pipeline NULL→PLAYING to
    // re-acquire the camera sources and restart streaming. Thread-safe w.r.t. the main loop.
    // Some stall modes need a full rebuild; this handles the common re-negotiable cases and,
    // regardless, is paired with a health flag so the robot degrades gracefully.
    void restart();

private:
    std::vector<CameraConfig> cameras_;
    std::string               infer_config_path_;
    DetectionCallback         on_detection_;
    GstElement*               pipeline_  = nullptr;
    GMainLoop*                loop_      = nullptr;

    static gboolean bus_cb(GstBus*, GstMessage*, gpointer);
};
