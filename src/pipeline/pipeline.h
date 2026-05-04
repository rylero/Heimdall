#pragma once
#include "camera_source.h"
#include "probe.h"
#include <gst/gst.h>
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

    // Set by on_media_configure; read by the q_rtsp pad probe.
    GstElement* rtsp_appsrc_ = nullptr;
    GMutex      rtsp_appsrc_mutex_{};

private:
    std::vector<CameraConfig> cameras_;
    std::string               infer_config_path_;
    DetectionCallback         on_detection_;
    GstElement*               pipeline_    = nullptr;
    GMainLoop*                loop_        = nullptr;
    GstRTSPServer*            rtsp_server_ = nullptr;

    static void     on_media_configure(GstRTSPMediaFactory*, GstRTSPMedia*, gpointer);
    static gboolean bus_cb(GstBus*, GstMessage*, gpointer);
};
