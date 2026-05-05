#include "pipeline.h"
#include <stdexcept>
#include <string>

static constexpr int RTSP_SERV_PORT = 8555;

DeepStreamPipeline::DeepStreamPipeline(
    std::vector<CameraConfig> cameras,
    std::string               infer_config_path,
    DetectionCallback         on_detection
) : cameras_(std::move(cameras)),
    infer_config_path_(std::move(infer_config_path)),
    on_detection_(std::move(on_detection))
{
    g_mutex_init(&rtsp_appsrc_mutex_);
}

DeepStreamPipeline::~DeepStreamPipeline() {
    stop();
    g_mutex_clear(&rtsp_appsrc_mutex_);
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

        GstElement* tee       = gst_element_factory_make("tee",      ("tee_" + std::to_string(i)).c_str());
        GstElement* q_infer   = gst_element_factory_make("queue",    ("qi_"  + std::to_string(i)).c_str());
        GstElement* conv_nvmm = gst_element_factory_make("nvvidconv",("cnv_" + std::to_string(i)).c_str());
        GstElement* q_rtsp    = gst_element_factory_make("queue",    ("qr_"  + std::to_string(i)).c_str());
        if (!tee || !q_infer || !conv_nvmm || !q_rtsp)
            throw std::runtime_error("Failed to create tee/queue/nvvidconv");
        g_object_set(q_infer, "leaky", 2, "max-size-buffers", 2, nullptr);
        g_object_set(q_rtsp,  "leaky", 2, "max-size-buffers", 2, nullptr);
        gst_bin_add_many(GST_BIN(pipeline_), tee, q_infer, conv_nvmm, q_rtsp, nullptr);

        GstPad* src_pad  = gst_element_get_static_pad(src, "src");
        GstPad* tee_sink = gst_element_get_static_pad(tee, "sink");
        if (gst_pad_link(src_pad, tee_sink) != GST_PAD_LINK_OK)
            throw std::runtime_error("Failed to link src to tee");
        gst_object_unref(src_pad); gst_object_unref(tee_sink);

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

        GstPad* t1 = gst_element_get_request_pad(tee, "src_%u");
        GstPad* qr = gst_element_get_static_pad(q_rtsp, "sink");
        gst_pad_link(t1, qr); gst_object_unref(t1); gst_object_unref(qr);

        if (i == 0) {
            // USB cameras exit the tee as system memory (jpegdec output) — nvvidconv
            // needs NVMM input and will stall on system memory, causing zero flow and
            // NULL current_caps on the probe pad.  Use videoconvert for USB (CPU→CPU)
            // and nvvidconv for CSI (NVMM→system).
            const bool is_csi = (cameras_[0].type == CameraType::CSI);
            GstElement* conv_rtsp = gst_element_factory_make(
                is_csi ? "nvvidconv" : "videoconvert", "rtsp_conv");
            GstElement* cap_rtsp  = gst_element_factory_make("capsfilter", "rtsp_cap");
            GstElement* fsink     = gst_element_factory_make("fakesink",   "rtsp_fsink");
            if (!conv_rtsp || !cap_rtsp || !fsink)
                throw std::runtime_error("Failed to create RTSP branch elements");

            GstCaps* rtsp_caps = gst_caps_new_simple("video/x-raw",
                "format", G_TYPE_STRING, "NV12",
                "width",  G_TYPE_INT,    static_cast<gint>(cameras_[0].width),
                "height", G_TYPE_INT,    static_cast<gint>(cameras_[0].height),
                nullptr);
            g_object_set(cap_rtsp, "caps", rtsp_caps, nullptr);
            gst_caps_unref(rtsp_caps);

            g_object_set(fsink, "sync", FALSE, nullptr);
            gst_bin_add_many(GST_BIN(pipeline_), conv_rtsp, cap_rtsp, fsink, nullptr);
            if (!gst_element_link_many(q_rtsp, conv_rtsp, cap_rtsp, fsink, nullptr))
                throw std::runtime_error("Failed to link rtsp branch");

            GstPad* qr_src = gst_element_get_static_pad(cap_rtsp, "src");
            gst_pad_add_probe(qr_src, GST_PAD_PROBE_TYPE_BUFFER,
                [](GstPad* pad, GstPadProbeInfo* info, gpointer data) -> GstPadProbeReturn {
                    auto* self = static_cast<DeepStreamPipeline*>(data);

                    g_mutex_lock(&self->rtsp_appsrc_mutex_);
                    GstElement* appsrc = self->rtsp_appsrc_
                        ? GST_ELEMENT(gst_object_ref(self->rtsp_appsrc_)) : nullptr;
                    g_mutex_unlock(&self->rtsp_appsrc_mutex_);

                    if (appsrc) {
                        GstBuffer* buf  = gst_buffer_ref(GST_PAD_PROBE_INFO_BUFFER(info));
                        GstCaps*   caps = gst_pad_get_current_caps(pad);
                        if (caps) {
                            GstSample* s = gst_sample_new(buf, caps, nullptr, nullptr);
                            gst_app_src_push_sample(GST_APP_SRC(appsrc), s);
                            gst_sample_unref(s);
                            gst_caps_unref(caps);
                        }
                        gst_buffer_unref(buf);
                        gst_object_unref(appsrc);
                    }
                    return GST_PAD_PROBE_OK;
                }, this, nullptr);
            gst_object_unref(qr_src);
        } else {
            GstElement* drop = gst_element_factory_make("fakesink",
                ("rtsp_drop_" + std::to_string(i)).c_str());
            gst_bin_add(GST_BIN(pipeline_), drop);
            gst_element_link(q_rtsp, drop);
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

    // RTSP server — mirrors the user's simple example:
    //   videotestsrc is-live=1 ! x264enc ! rtph264pay name=pay0 pt=96
    // but with appsrc instead of videotestsrc, fed by the q_rtsp pad probe.
    // is-live=1 → NO_PREROLL → prepare() sets PLAYING immediately →
    // first real frame arrives via probe → caps flow → check_prerolled → 200 OK.
    const std::string w   = std::to_string(cameras_[0].width);
    const std::string h   = std::to_string(cameras_[0].height);
    const std::string fps = std::to_string(cameras_[0].fps);
    // Inline caps after appsrc → GStreamer resolves SDP at DESCRIBE time
    // without waiting for a client to push a frame first.
    const std::string factory_str =
        "( appsrc name=rtsp_src is-live=true format=time "
        "! video/x-raw,format=NV12,width=" + w + ",height=" + h + ",framerate=" + fps + "/1 "
        "! videoconvert "
        "! video/x-raw,format=I420 "
        "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 "
        "! rtph264pay name=pay0 pt=96 )";
    rtsp_server_ = gst_rtsp_server_new();
    gst_rtsp_server_set_service(rtsp_server_, std::to_string(RTSP_SERV_PORT).c_str());

    GstRTSPMountPoints*  mounts  = gst_rtsp_server_get_mount_points(rtsp_server_);
    GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, factory_str.c_str());
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    g_signal_connect(factory, "media-configure",
        G_CALLBACK(on_media_configure), this);
    gst_rtsp_mount_points_add_factory(mounts, "/stream", factory);
    g_object_unref(mounts);
    gst_rtsp_server_attach(rtsp_server_, nullptr);

    std::printf("RTSP stream: rtsp://0.0.0.0:%d/stream\n", RTSP_SERV_PORT);
}

// Store the factory's appsrc so the q_rtsp pad probe can push frames into it.
void DeepStreamPipeline::on_media_configure(GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer data) {
    auto* self = static_cast<DeepStreamPipeline*>(data);
    gst_rtsp_media_set_suspend_mode(media, GST_RTSP_SUSPEND_MODE_NONE);
    GstElement* bin = gst_rtsp_media_get_element(media);
    GstElement* appsrc = gst_bin_get_by_name(GST_BIN(bin), "rtsp_src");
    gst_object_unref(bin);

    if (appsrc) {
        // Set caps now so the appsrc reports a non-NULL format immediately —
        // the RTSP server queries get_caps during SDP construction and the
        // probe may call gst_pad_get_current_caps before the first buffer flows.
        GstCaps* caps = gst_caps_new_simple("video/x-raw",
            "format",    G_TYPE_STRING,       "NV12",
            "width",     G_TYPE_INT,          static_cast<gint>(self->cameras_[0].width),
            "height",    G_TYPE_INT,          static_cast<gint>(self->cameras_[0].height),
            "framerate", GST_TYPE_FRACTION,   self->cameras_[0].fps, 1,
            nullptr);
        gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
        gst_caps_unref(caps);

        g_object_set(G_OBJECT(appsrc), "is-live", TRUE, "min-latency", (gint64)0, nullptr);

        g_mutex_lock(&self->rtsp_appsrc_mutex_);
        if (self->rtsp_appsrc_) gst_object_unref(self->rtsp_appsrc_);
        self->rtsp_appsrc_ = GST_ELEMENT(gst_object_ref(appsrc));
        g_mutex_unlock(&self->rtsp_appsrc_mutex_);
        gst_object_unref(appsrc);
    }
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
    g_mutex_lock(&rtsp_appsrc_mutex_);
    if (rtsp_appsrc_) { gst_object_unref(rtsp_appsrc_); rtsp_appsrc_ = nullptr; }
    g_mutex_unlock(&rtsp_appsrc_mutex_);
}