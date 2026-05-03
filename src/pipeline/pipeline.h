#pragma once
#include "camera_source.h"
#include "probe.h"
#include <functional>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <string>
#include <vector>

using FrameCallback = std::function<void(const uint8_t*, size_t)>;

class DeepStreamPipeline {
public:
    DeepStreamPipeline(
        std::vector<CameraConfig> cameras,
        std::string               infer_config_path,
        DetectionCallback         on_detection,
        FrameCallback             on_frame = {}
    );
    ~DeepStreamPipeline();

    void build();
    void run();    // blocks until stop() or error
    void stop();

private:
    std::vector<CameraConfig> cameras_;
    std::string               infer_config_path_;
    DetectionCallback         on_detection_;
    FrameCallback             on_frame_;
    GstElement*               pipeline_ = nullptr;
    GMainLoop*                loop_     = nullptr;

    static gboolean      bus_cb(GstBus*, GstMessage*, gpointer);
    static GstFlowReturn appsink_cb(GstAppSink*, gpointer);
};
