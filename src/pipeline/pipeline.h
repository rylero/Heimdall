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
    void run();    // blocks until stop() or error
    void stop();

private:
    std::vector<CameraConfig> cameras_;
    std::string               infer_config_path_;
    DetectionCallback         on_detection_;
    GstElement*               pipeline_ = nullptr;
    GMainLoop*                loop_     = nullptr;

    static gboolean bus_cb(GstBus*, GstMessage*, gpointer);
};
