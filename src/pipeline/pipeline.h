#pragma once
#include "camera_source.h"
#include "probe.h"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
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

private:
    std::vector<CameraConfig> cameras_;
    std::string               infer_config_path_;
    DetectionCallback         on_detection_;
    GstElement*               pipeline_    = nullptr;
    GMainLoop*                loop_        = nullptr;
    GstRTSPServer*            rtsp_server_ = nullptr;

    // appsink in main pipeline captures raw frames; appsrc in RTSP factory
    // consumes them so the factory pipeline can encode independently.
    GstElement*               rtsp_appsink_ = nullptr;
    GstElement*               rtsp_appsrc_  = nullptr;
    GMutex                    rtsp_appsrc_mutex_{};

    static gboolean      bus_cb(GstBus*, GstMessage*, gpointer);
    static void          on_media_configure(GstRTSPMediaFactory*, GstRTSPMedia*, gpointer);
    static GstFlowReturn on_new_sample(GstAppSink*, gpointer);
};
