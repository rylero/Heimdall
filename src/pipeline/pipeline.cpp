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

    // RTSP output elements (system-memory path — split before NVMM conversion)
    GstElement* rtsp_vcvt   = gst_element_factory_make("videoconvert", "rtsp_vcvt");
    GstElement* rtsp_enc    = gst_element_factory_make("x264enc",      "rtsp_enc");
    GstElement* rtsp_pay    = gst_element_factory_make("rtph264pay",   "rtsp_pay");
    GstElement* rtsp_sink   = gst_element_factory_make("udpsink",      "rtsp_udp");
    if (!rtsp_vcvt || !rtsp_enc || !rtsp_pay || !rtsp_sink)
        throw std::runtime_error("Failed to create RTSP elements");
    g_object_set(rtsp_enc,  "bitrate", 4000, "tune", 4, nullptr);
    g_object_set(rtsp_pay,  "config-interval", 1, "pt", 96, nullptr);
    g_object_set(rtsp_sink, "host", "127.0.0.1", "port", RTSP_UDP_PORT, "sync", FALSE, nullptr);
    gst_bin_add_many(GST_BIN(pipeline_), rtsp_vcvt, rtsp_enc, rtsp_pay, rtsp_sink, nullptr);

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

        // Tee after jpegdec (system memory) so RTSP branch avoids NVMM entirely
        GstElement* tee       = gst_element_factory_make("tee",      ("tee_"  + std::to_string(i)).c_str());
        GstElement* q_infer   = gst_element_factory_make("queue",    ("qi_"   + std::to_string(i)).c_str());
        GstElement* conv_nvmm = gst_element_factory_make("nvvidconv",("cnv_"  + std::to_string(i)).c_str());
        GstElement* q_rtsp    = gst_element_factory_make("queue",    ("qr_"   + std::to_string(i)).c_str());
        if (!tee || !q_infer || !conv_nvmm || !q_rtsp)
            throw std::runtime_error("Failed to create tee/queue/nvvidconv for src " + std::to_string(i));
        // Leaky queues: if one branch stalls (e.g. NVMM allocation fails), the other keeps running
        g_object_set(q_infer, "leaky", 2, "max-size-buffers", 2, nullptr);
        g_object_set(q_rtsp,  "leaky", 2, "max-size-buffers", 2, nullptr);
        gst_bin_add_many(GST_BIN(pipeline_), tee, q_infer, conv_nvmm, q_rtsp, nullptr);

        // src (jpegdec output, system memory) → tee
        GstPad* src_pad  = gst_element_get_static_pad(src, "src");
        GstPad* tee_sink = gst_element_get_static_pad(tee, "sink");
        if (gst_pad_link(src_pad, tee_sink) != GST_PAD_LINK_OK)
            throw std::runtime_error("Failed to link src to tee");
        gst_object_unref(src_pad); gst_object_unref(tee_sink);

        // Tee branch 0: q_infer → nvvidconv (NVMM) → mux
        GstPad* t0 = gst_element_get_request_pad(tee, "src_%u");
        GstPad* qi = gst_element_get_static_pad(q_infer, "sink");
        gst_pad_link(t0, qi); gst_object_unref(t0); gst_object_unref(qi);
        if (!gst_element_link(q_infer, conv_nvmm))
            throw std::runtime_error("Failed to link q_infer to nvvidconv");
        GstPad* conv_src = gst_element_get_static_pad(conv_nvmm, "src");
        GstPad* mux_sink = gst_element_get_request_pad(mux, ("sink_" + std::to_string(i)).c_str());
        if (gst_pad_link(conv_src, mux_sink) != GST_PAD_LINK_OK)
            throw std::runtime_error("Failed to link nvvidconv to mux");
        gst_object_unref(conv_src); gst_object_unref(mux_sink);

        // Tee branch 1 (camera 0 only): q_rtsp → videoconvert → x264enc chain
        GstPad* t1 = gst_element_get_request_pad(tee, "src_%u");
        GstPad* qr = gst_element_get_static_pad(q_rtsp, "sink");
        gst_pad_link(t1, qr); gst_object_unref(t1); gst_object_unref(qr);
        if (i == 0) {
            if (!gst_element_link_many(q_rtsp, rtsp_vcvt, rtsp_enc, rtsp_pay, rtsp_sink, nullptr))
                throw std::runtime_error("Failed to link RTSP branch");

            // Debug probes — remove once RTSP is confirmed working
            auto count_probe = [](GstPad*, GstPadProbeInfo*, gpointer label) -> GstPadProbeReturn {
                static int n = 0;
                if (++n % 90 == 1) g_print("RTSP probe [%s]: %d buffers\n", (const char*)label, n);
                return GST_PAD_PROBE_OK;
            };
            GstPad* pq = gst_element_get_static_pad(q_rtsp,   "src"); gst_pad_add_probe(pq, GST_PAD_PROBE_TYPE_BUFFER, count_probe, (gpointer)"q_rtsp",   nullptr); gst_object_unref(pq);
            GstPad* pe = gst_element_get_static_pad(rtsp_enc,  "src"); gst_pad_add_probe(pe, GST_PAD_PROBE_TYPE_BUFFER, count_probe, (gpointer)"x264enc",  nullptr); gst_object_unref(pe);
            GstPad* pp = gst_element_get_static_pad(rtsp_pay,  "src"); gst_pad_add_probe(pp, GST_PAD_PROBE_TYPE_BUFFER, count_probe, (gpointer)"rtph264pay",nullptr); gst_object_unref(pp);
        } else {
            GstElement* rtsp_drop = gst_element_factory_make("fakesink", ("rtsp_drop_" + std::to_string(i)).c_str());
            gst_bin_add(GST_BIN(pipeline_), rtsp_drop);
            gst_element_link(q_rtsp, rtsp_drop);
        }
    }

    GstElement* infer = gst_element_factory_make("nvinfer", "infer");
    if (!infer) throw std::runtime_error("Failed to create nvinfer");
    g_object_set(infer, "config-file-path", infer_config_path_.c_str(), nullptr);
    gst_bin_add(GST_BIN(pipeline_), infer);

    GstElement* fakesink = gst_element_factory_make("fakesink", "sink");
    if (!fakesink) throw std::runtime_error("Failed to create fakesink");
    g_object_set(fakesink, "sync", FALSE, nullptr);
    gst_bin_add(GST_BIN(pipeline_), fakesink);

    if (!gst_element_link(mux, infer) || !gst_element_link(infer, fakesink))
        throw std::runtime_error("Failed to link mux→infer→fakesink");

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
        "( udpsrc port=5400 "
        "caps=\"application/x-rtp,media=video,clock-rate=90000,"
        "encoding-name=H264,payload=96\" "
        "! rtph264depay ! rtph264pay name=pay0 pt=96 )");
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
