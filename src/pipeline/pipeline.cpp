#include "pipeline.h"
#include <stdexcept>
#include <string>

static constexpr int RTSP_UDP_PORT  = 5400;
static constexpr int RTSP_SERV_PORT = 8554;

DeepStreamPipeline::DeepStreamPipeline(
    std::vector<CameraConfig> cameras,
    std::string               infer_config_path,
    DetectionCallback         on_detection
) : cameras_(std::move(cameras)),
    infer_config_path_(std::move(infer_config_path)),
    on_detection_(std::move(on_detection)) {}

DeepStreamPipeline::~DeepStreamPipeline() { stop(); }

void DeepStreamPipeline::build() {
    gst_init(nullptr, nullptr);

    pipeline_ = gst_pipeline_new("heimdall");
    if (!pipeline_) throw std::runtime_error("Failed to create pipeline");

    GstElement* mux = gst_element_factory_make("nvstreammux", "mux");
    if (!mux) throw std::runtime_error("Failed to create nvstreammux — DeepStream plugins not loaded?");
    g_object_set(mux,
        "width",                static_cast<gint>(cameras_[0].width),
        "height",               static_cast<gint>(cameras_[0].height),
        "batch-size",           static_cast<gint>(cameras_.size()),
        "batched-push-timeout", 4000000,
        "live-source",          TRUE,
        nullptr);
    gst_bin_add(GST_BIN(pipeline_), mux);

    for (int i = 0; i < static_cast<int>(cameras_.size()); ++i) {
        GError* err = nullptr;
        GstElement* src = gst_parse_bin_from_description(
            build_source_description(cameras_[i]).c_str(), TRUE, &err);
        if (!src) {
            std::string msg = err ? err->message : "unknown error";
            g_clear_error(&err);
            throw std::runtime_error("Failed to create source bin: " + msg);
        }
        gst_element_set_name(src, ("src_" + std::to_string(i)).c_str());
        gst_bin_add(GST_BIN(pipeline_), src);

        GstPad* src_pad  = gst_element_get_static_pad(src, "src");
        GstPad* sink_pad = gst_element_get_request_pad(mux,
            ("sink_" + std::to_string(i)).c_str());
        if (gst_pad_link(src_pad, sink_pad) != GST_PAD_LINK_OK)
            throw std::runtime_error("Failed to link src_" + std::to_string(i) + " to mux");
        gst_object_unref(src_pad);
        gst_object_unref(sink_pad);
    }

    GstElement* infer = gst_element_factory_make("nvinfer", "infer");
    if (!infer) throw std::runtime_error("Failed to create nvinfer");
    g_object_set(infer, "config-file-path", infer_config_path_.c_str(), nullptr);
    gst_bin_add(GST_BIN(pipeline_), infer);

    // Draw detection overlays on the frame
    GstElement* osd = gst_element_factory_make("nvdsosd", "osd");
    if (!osd) throw std::runtime_error("Failed to create nvdsosd");
    gst_bin_add(GST_BIN(pipeline_), osd);

    // RTSP output chain: nvvideoconvert → avenc_h264 → rtph264pay → udpsink
    GstElement* conv    = gst_element_factory_make("nvvideoconvert", "rtsp_conv");
    GstElement* enc     = gst_element_factory_make("avenc_h264",     "rtsp_enc");
    GstElement* pay     = gst_element_factory_make("rtph264pay",     "rtsp_pay");
    GstElement* udpsink = gst_element_factory_make("udpsink",        "rtsp_udp");
    if (!conv)    throw std::runtime_error("Failed to create nvvideoconvert");
    if (!enc)     throw std::runtime_error("Failed to create avenc_h264 — gstreamer1.0-libav missing?");
    if (!pay)     throw std::runtime_error("Failed to create rtph264pay");
    if (!udpsink) throw std::runtime_error("Failed to create udpsink");

    g_object_set(enc,     "bitrate", 4000000, nullptr);
    g_object_set(pay,     "config-interval", 1, "pt", 96, nullptr);
    g_object_set(udpsink, "host", "127.0.0.1", "port", RTSP_UDP_PORT,
                          "sync", FALSE, nullptr);
    gst_bin_add_many(GST_BIN(pipeline_), conv, enc, pay, udpsink, nullptr);

    if (!gst_element_link(mux, infer))
        throw std::runtime_error("Failed to link mux to infer");
    if (!gst_element_link_many(infer, osd, conv, enc, pay, udpsink, nullptr))
        throw std::runtime_error("Failed to link RTSP chain");

    // Probe on infer src to extract detections (before osd renders boxes)
    GstPad* infer_src = gst_element_get_static_pad(infer, "src");
    gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER,
        detection_probe_cb, &on_detection_, nullptr);
    gst_object_unref(infer_src);

    GstBus* bus = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus, bus_cb, this);
    gst_object_unref(bus);

    // RTSP server: mounts a udpsrc pointing at the H.264 RTP stream
    rtsp_server_ = gst_rtsp_server_new();
    gst_rtsp_server_set_service(rtsp_server_, std::to_string(RTSP_SERV_PORT).c_str());

    GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(rtsp_server_);
    GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory,
        "( udpsrc name=pay0 port=5400 "
        "caps=\"application/x-rtp,media=video,clock-rate=90000,"
        "encoding-name=H264,payload=96\" )");
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_mount_points_add_factory(mounts, "/stream", factory);
    g_object_unref(mounts);
    gst_rtsp_server_attach(rtsp_server_, nullptr);

    std::printf("RTSP stream: rtsp://0.0.0.0:%d/stream\n", RTSP_SERV_PORT);
}

gboolean DeepStreamPipeline::bus_cb(GstBus*, GstMessage* msg, gpointer data) {
    auto* self = static_cast<DeepStreamPipeline*>(data);
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError* err; gchar* dbg;
            gst_message_parse_error(msg, &err, &dbg);
            g_printerr("Pipeline error: %s\n%s\n", err->message, dbg ? dbg : "");
            g_error_free(err); g_free(dbg);
            if (self->loop_) g_main_loop_quit(self->loop_);
            break;
        }
        case GST_MESSAGE_EOS:
            if (self->loop_) g_main_loop_quit(self->loop_);
            break;
        default: break;
    }
    return TRUE;
}

void DeepStreamPipeline::run() {
    if (!pipeline_) build();
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    loop_ = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(loop_);
}

void DeepStreamPipeline::stop() {
    if (loop_) {
        g_main_loop_quit(loop_);
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    if (rtsp_server_) {
        g_object_unref(rtsp_server_);
        rtsp_server_ = nullptr;
    }
}
