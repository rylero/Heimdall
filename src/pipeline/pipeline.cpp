#include "pipeline.h"
#include <cmath>
#include <stdexcept>
#include <string>

static constexpr int RTSP_SERV_PORT = 8554;

DeepStreamPipeline::DeepStreamPipeline(
    std::vector<CameraConfig> cameras,
    std::string               infer_config_path,
    DetectionCallback         on_detection
) : cameras_(std::move(cameras)),
    infer_config_path_(std::move(infer_config_path)),
    on_detection_(std::move(on_detection))
{}

DeepStreamPipeline::~DeepStreamPipeline() {
    stop();
}

void DeepStreamPipeline::build() {
    gst_init(nullptr, nullptr);

    pipeline_ = gst_pipeline_new("heimdall");
    if (!pipeline_) throw std::runtime_error("Failed to create pipeline");

    GstElement* mux = gst_element_factory_make("nvstreammux", "mux");
    if (!mux) throw std::runtime_error("Failed to create nvstreammux");
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
        if (!src_pad) throw std::runtime_error("Source bin has no 'src' ghost pad for camera " + std::to_string(i));
        GstPad* mux_sink = gst_element_get_request_pad(mux, ("sink_" + std::to_string(i)).c_str());
        if (!mux_sink) throw std::runtime_error("Failed to get mux sink pad for camera " + std::to_string(i));

        if (cameras_[i].type == CameraType::USB) {
            // USB sources output CPU memory after jpegdec; nvvidconv converts to NVMM
            GstElement* conv = gst_element_factory_make("nvvidconv",
                ("cnv_" + std::to_string(i)).c_str());
            if (!conv) throw std::runtime_error("Failed to create nvvidconv");
            gst_bin_add(GST_BIN(pipeline_), conv);

            GstPad* conv_sink = gst_element_get_static_pad(conv, "sink");
            if (gst_pad_link(src_pad, conv_sink) != GST_PAD_LINK_OK)
                throw std::runtime_error("Failed to link src to nvvidconv");
            gst_object_unref(conv_sink);

            GstPad* conv_src = gst_element_get_static_pad(conv, "src");
            if (gst_pad_link(conv_src, mux_sink) != GST_PAD_LINK_OK)
                throw std::runtime_error("Failed to link nvvidconv to mux");
            gst_object_unref(conv_src);
        } else {
            // CSI sources already output NVMM; link directly to mux
            if (gst_pad_link(src_pad, mux_sink) != GST_PAD_LINK_OK)
                throw std::runtime_error("Failed to link src to mux");
        }
        gst_object_unref(src_pad);
        gst_object_unref(mux_sink);
    }

    GstElement* infer = gst_element_factory_make("nvinfer", "infer");
    if (!infer) throw std::runtime_error("Failed to create nvinfer");
    g_object_set(infer, "config-file-path", infer_config_path_.c_str(), nullptr);
    gst_bin_add(GST_BIN(pipeline_), infer);

    // Probe fires before tiling so frame->source_id still maps to original camera index
    GstPad* infer_src = gst_element_get_static_pad(infer, "src");
    gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER,
        detection_probe_cb, &on_detection_, nullptr);
    gst_object_unref(infer_src);

    guint tiler_rows = static_cast<guint>(std::ceil(std::sqrt(cameras_.size())));
    guint tiler_cols = static_cast<guint>(
        std::ceil(static_cast<double>(cameras_.size()) / tiler_rows));
    GstElement* tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
    if (!tiler) throw std::runtime_error("Failed to create nvmultistreamtiler");
    g_object_set(tiler,
        "rows",    tiler_rows,
        "columns", tiler_cols,
        "width",   cameras_[0].width  * tiler_cols,
        "height",  cameras_[0].height * tiler_rows,
        nullptr);
    gst_bin_add(GST_BIN(pipeline_), tiler);

    GstElement* osd = gst_element_factory_make("nvdsosd", "osd");
    if (!osd) throw std::runtime_error("Failed to create nvdsosd");
    // process-mode=1 (GPU/CUDA) works headless — no EGL/display context needed.
    // process-mode=0 (CPU) also works headless but is slower; mode=2 (HW/VIC) is Jetson-only.
    g_object_set(osd, "process-mode", 1, nullptr);
    gst_bin_add(GST_BIN(pipeline_), osd);

    // nvvidconv: convert NVMM output from nvdsosd to the format the encoder expects
    GstElement* conv_out = gst_element_factory_make("nvvidconv", "conv_out");
    if (!conv_out) throw std::runtime_error("Failed to create nvvidconv");
    gst_bin_add(GST_BIN(pipeline_), conv_out);

    // Choose encoder based on HW availability
    GstElement* test_enc = gst_element_factory_make("nvv4l2h264enc", nullptr);
    const bool  hw_enc   = (test_enc != nullptr);
    if (test_enc) {
        gst_object_unref(test_enc);
    } else {
        g_printerr("[pipeline] nvv4l2h264enc unavailable, using x264enc\n");
    }

    GstElement* encoder;
    GstCaps*    enc_caps;
    if (hw_enc) {
        encoder  = gst_element_factory_make("nvv4l2h264enc", "encoder");
        if (!encoder) throw std::runtime_error("Failed to create nvv4l2h264enc");
        g_object_set(encoder, "bitrate", static_cast<guint>(4000000), nullptr);
        enc_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
    } else {
        encoder  = gst_element_factory_make("x264enc", "encoder");
        if (!encoder) throw std::runtime_error("No H264 encoder available");
        g_object_set(encoder, "bitrate", 4000u, nullptr);  // kbps
        enc_caps = gst_caps_from_string("video/x-raw,format=I420");
    }
    gst_bin_add(GST_BIN(pipeline_), encoder);

    GstElement* caps_out = gst_element_factory_make("capsfilter", "caps_out");
    if (!caps_out) throw std::runtime_error("Failed to create capsfilter");
    g_object_set(caps_out, "caps", enc_caps, nullptr);
    gst_caps_unref(enc_caps);
    gst_bin_add(GST_BIN(pipeline_), caps_out);

    GstElement* rtp_pay = gst_element_factory_make("rtph264pay", "rtp_pay");
    if (!rtp_pay) throw std::runtime_error("Failed to create rtph264pay");
    g_object_set(rtp_pay, "config-interval", 1, "pt", 96, nullptr);
    gst_bin_add(GST_BIN(pipeline_), rtp_pay);

    // rtspclientsink publishes the encoded H264 stream to MediaMTX via RTSP ANNOUNCE.
    // MediaMTX (sidecar Docker service) serves it to viewers — no GstRTSPServer
    // linking in our binary, no preroll issues, works with any encoder.
    std::string rtsp_url = "rtsp://127.0.0.1:" + std::to_string(RTSP_SERV_PORT) + "/ds-test";
    GstElement* rtsp_sink = gst_element_factory_make("rtspclientsink", "rtsp_sink");
    if (!rtsp_sink) throw std::runtime_error("Failed to create rtspclientsink");
    g_object_set(rtsp_sink, "location", rtsp_url.c_str(), nullptr);
    gst_bin_add(GST_BIN(pipeline_), rtsp_sink);

    if (!gst_element_link(mux, infer))
        throw std::runtime_error("Failed to link mux→infer");
    if (!gst_element_link(infer, tiler))
        throw std::runtime_error("Failed to link infer→tiler");
    if (!gst_element_link(tiler, osd))
        throw std::runtime_error("Failed to link tiler→osd");
    if (!gst_element_link(osd, conv_out))
        throw std::runtime_error("Failed to link osd→conv_out");
    if (!gst_element_link(conv_out, caps_out))
        throw std::runtime_error("Failed to link conv_out→caps_out");
    if (!gst_element_link(caps_out, encoder))
        throw std::runtime_error("Failed to link caps_out→encoder");
    if (!gst_element_link(encoder, rtp_pay))
        throw std::runtime_error("Failed to link encoder→rtp_pay");
    // rtspclientsink uses request pads (sink_0, sink_1, ...) — must request explicitly
    {
        GstPad* pay_src  = gst_element_get_static_pad(rtp_pay, "src");
        GstPad* sink_pad = gst_element_get_request_pad(rtsp_sink, "sink_0");
        if (!sink_pad) throw std::runtime_error("Failed to get rtspclientsink sink_0 pad");
        if (gst_pad_link(pay_src, sink_pad) != GST_PAD_LINK_OK)
            throw std::runtime_error("Failed to link rtp_pay→rtsp_sink");
        gst_object_unref(pay_src);
        gst_object_unref(sink_pad);
    }

    add_stage_probe(tiler,    "tiler");
    add_stage_probe(osd,      "osd");
    add_stage_probe(conv_out, "conv_out");
    add_stage_probe(encoder,  "encoder");

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline_), GST_DEBUG_GRAPH_SHOW_ALL, "heimdall-pipeline");

    GstBus* bus = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus, bus_cb, this);
    gst_object_unref(bus);

    std::printf("RTSP stream: rtsp://0.0.0.0:%d/ds-test  (served by MediaMTX)\n", RTSP_SERV_PORT);
}

GstPadProbeReturn DeepStreamPipeline::stage_probe_cb(GstPad*, GstPadProbeInfo*, gpointer data) {
    auto* sc = static_cast<StageCounter*>(data);
    ++sc->count;
    if (sc->count <= 10 || sc->count % 100 == 0)
        g_printerr("[stage] %s frame %d\n", sc->name, sc->count);
    return GST_PAD_PROBE_OK;
}

void DeepStreamPipeline::add_stage_probe(GstElement* element, const char* stage_name) {
    stage_counters_.push_back({stage_name, 0});
    StageCounter* sc = &stage_counters_.back();
    GstPad* src_pad = gst_element_get_static_pad(element, "src");
    if (!src_pad) {
        g_printerr("[stage] no src pad on %s — skipping probe\n", stage_name);
        return;
    }
    gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER, stage_probe_cb, sc, nullptr);
    gst_object_unref(src_pad);
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
}
