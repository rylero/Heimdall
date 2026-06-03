#pragma once
#include "camera_source.h"
#include "probe.h"
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <list>
#include <string>
#include <vector>

struct StageCounter {
    const char* name;
    int         count = 0;
};

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
    GstElement*               pipeline_       = nullptr;
    GMainLoop*                loop_           = nullptr;
    GstRTSPServer*            rtsp_server_    = nullptr;
    guint                     rtsp_source_id_ = 0;
    std::list<StageCounter>   stage_counters_;

    void add_stage_probe(GstElement* element, const char* stage_name);
    static GstPadProbeReturn stage_probe_cb(GstPad*, GstPadProbeInfo*, gpointer);
    static gboolean bus_cb(GstBus*, GstMessage*, gpointer);
    static void media_configure_cb(GstRTSPMediaFactory*, GstRTSPMedia*, gpointer);
};
