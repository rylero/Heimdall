#include "pipeline.h"
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

        // Tee after jpegdec (system memory) so RTSP branch avoids NVMM entirely
        GstElement* tee       = gst_element_factory_make("tee",      ("tee_"  + std::to_string(i)).c_str());
        GstElement* q_infer   = gst_element_factory_make("queue",    ("qi_"   + std::to_string(i)).c_str());
        GstElement* conv_nvmm = gst_element_factory_make("nvvidconv",("cnv_"  + std::to_string(i)).c_str());
        GstElement* q_rtsp    = gst_element_factory_make("queue",    ("qr_"   + std::to_string(i)).c_str());
        if (!tee || !q_infer || !conv_nvmm || !q_rtsp)
            throw std::runtime_error("Failed to create tee/queue/nvvidconv for src " + std::to_string(i));
        g_object_set(q_infer, "leaky", 2, "max-size-buffers", 2, nullptr);
        g_object_set(q_rtsp,  "leaky", 2, "max-size-buffers", 2, nullptr);
        gst_bin_add_many(GST_BIN(pipeline_), tee, q_infer, conv_nvmm, q_rtsp, nullptr);

        GstPad* src_pad  = gst_element_get_static_pad(src, "src");
        GstPad* tee_sink = gst_element_get_static_pad(tee, "sink");
        if (gst_pad_link(src_pad, tee_sink) != GST_PAD_LINK_OK)
            throw std::runtime_error("Failed to link src to tee");
        gst_object_unref(src_pad); gst_object_unref(tee_sink);

        // Tee branch 0: inference path → NVMM → mux
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

        // Tee branch 1: RTSP path — raw frames captured by appsink for the factory
        GstPad* t1 = gst_element_get_request_pad(tee, "src_%u");
        GstPad* qr = gst_element_get_static_pad(q_rtsp, "sink");
        gst_pad_link(t1, qr); gst_object_unref(t1); gst_object_unref(qr);

        if (i == 0) {
            GstElement* vcvt   = gst_element_factory_make("videoconvert", "rtsp_vcvt");
            GstElement* cfilt  = gst_element_factory_make("capsfilter",   "rtsp_cfilt");
            GstElement* asink  = gst_element_factory_make("appsink",      "rtsp_appsink");
            if (!vcvt || !cfilt || !asink)
                throw std::runtime_error("Failed to create RTSP appsink elements");

            GstCaps* i420 = gst_caps_from_string("video/x-raw,format=I420");
            g_object_set(cfilt, "caps", i420, nullptr);
            gst_caps_unref(i420);

            g_object_set(asink,
                "emit-signals", TRUE,
                "sync",         FALSE,
                "max-buffers",  2,
                "drop",         TRUE,
                nullptr);

            gst_bin_add_many(GST_BIN(pipeline_), vcvt, cfilt, asink, nullptr);
            if (!gst_element_link_many(q_rtsp, vcvt, cfilt, asink, nullptr))
                throw std::runtime_error("Failed to link RTSP appsink branch");

            rtsp_appsink_ = asink;
            g_signal_connect(asink, "new-sample", G_CALLBACK(on_new_sample), this);
        } else {
            GstElement* drop = gst_element_factory_make("fakesink", ("rtsp_drop_" + std::to_string(i)).c_str());
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

    // RTSP server: factory pipeline encodes independently using appsrc fed by
    // the main pipeline's appsink.  Encoding in the factory means rtph264pay
    // has fully negotiated caps at PAUSED time, so gst_rtsp_media_prepare()
    // can generate a valid SDP before any data flows.
    const std::string w = std::to_string(cameras_[0].width);
    const std::string h = std::to_string(cameras_[0].height);
    // is-live=false so prepare() prerolls the pipeline and negotiates caps
    // before building the SDP.  fakesink after pay0 provides a downstream
    // sink so preroll completes; gst-rtsp-server unlinks it when a client
    // sends SETUP and inserts its own transport element instead.
    // format=3 (GST_FORMAT_TIME) prevents the segment-format CRITICAL.
    const std::string factory_str =
        "( appsrc name=rtsp_src is-live=false format=3 "
        "caps=\"video/x-raw,format=I420,width=" + w + ",height=" + h + "\" "
        "! x264enc tune=4 bitrate=4000 "
        "! rtph264pay name=pay0 config-interval=1 pt=96 "
        "! fakesink sync=false )";

    // Verify the factory pipeline parses before handing it to gst-rtsp-server.
    {
        std::string inner = factory_str.substr(1, factory_str.size() - 2);
        GError* perr = nullptr;
        GstElement* test = gst_parse_bin_from_description(inner.c_str(), TRUE, &perr);
        if (test) {
            std::printf("[RTSP] factory pipeline OK: %s\n", factory_str.c_str());
            gst_object_unref(test);
        } else {
            g_printerr("[RTSP] factory pipeline PARSE ERROR: %s\n  string: %s\n",
                perr ? perr->message : "unknown", factory_str.c_str());
            g_clear_error(&perr);
        }
    }

    rtsp_server_ = gst_rtsp_server_new();
    gst_rtsp_server_set_service(rtsp_server_, std::to_string(RTSP_SERV_PORT).c_str());

    GstRTSPMountPoints*  mounts  = gst_rtsp_server_get_mount_points(rtsp_server_);
    GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, factory_str.c_str());
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    g_signal_connect(factory, "media-configure", G_CALLBACK(on_media_configure), this);
    gst_rtsp_mount_points_add_factory(mounts, "/stream", factory);
    g_object_unref(mounts);
    gst_rtsp_server_attach(rtsp_server_, nullptr);

    std::printf("RTSP stream: rtsp://0.0.0.0:%d/stream\n", RTSP_SERV_PORT);
}

// Called when gst-rtsp-server constructs the shared media pipeline.
// Wires the factory's appsrc to the main pipeline's appsink and primes it
// with one black frame.  Because the factory uses is-live=false + fakesink,
// prepare() will preroll the pipeline (fakesink receives the black frame →
// ASYNC_DONE posted), negotiate caps on rtph264pay's src pad, and build a
// valid SDP — all without us manually managing pipeline state here.
void DeepStreamPipeline::on_media_configure(
    GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer data)
{
    auto* self = static_cast<DeepStreamPipeline*>(data);

    // Keep the prepared media alive when clients disconnect so subsequent
    // DESCRIBE requests reuse the already-prepared pipeline.
    gst_rtsp_media_set_suspend_mode(media, GST_RTSP_SUSPEND_MODE_NONE);

    GstElement* bin    = gst_rtsp_media_get_element(media);
    GstElement* appsrc = gst_bin_get_by_name(GST_BIN(bin), "rtsp_src");
    gst_object_unref(bin);

    if (!appsrc) {
        g_printerr("[RTSP] appsrc 'rtsp_src' not found\n");
        return;
    }

    g_mutex_lock(&self->rtsp_appsrc_mutex_);
    if (self->rtsp_appsrc_) gst_object_unref(self->rtsp_appsrc_);
    self->rtsp_appsrc_ = appsrc;
    g_mutex_unlock(&self->rtsp_appsrc_mutex_);

    // Prime the appsrc queue with one black I420 frame.  When prepare()
    // transitions the pipeline to PAUSED, this frame flows through
    // x264enc → rtph264pay → fakesink, completing preroll and negotiating
    // caps so gst-rtsp-server can build the SDP without any real camera frame.
    const gint w = static_cast<gint>(self->cameras_[0].width);
    const gint h = static_cast<gint>(self->cameras_[0].height);
    const gsize sz = static_cast<gsize>(w * h * 3 / 2);
    GstBuffer* buf = gst_buffer_new_allocate(nullptr, sz, nullptr);
    gst_buffer_memset(buf, 0, 0, sz);
    GST_BUFFER_PTS(buf)      = 0;
    GST_BUFFER_DTS(buf)      = 0;
    GST_BUFFER_DURATION(buf) = GST_SECOND / 30;
    GstCaps* vcaps = gst_caps_new_simple("video/x-raw",
        "format",    G_TYPE_STRING,    "I420",
        "width",     G_TYPE_INT,       w,
        "height",    G_TYPE_INT,       h,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        nullptr);
    GstSample* sample = gst_sample_new(buf, vcaps, nullptr, nullptr);
    gst_buffer_unref(buf);
    gst_caps_unref(vcaps);
    gst_app_src_push_sample(GST_APP_SRC(appsrc), sample);
    gst_sample_unref(sample);
    std::printf("[RTSP] factory configured, dummy frame queued\n");
}

// Called from the main pipeline's streaming thread each time a raw frame
// is available.  Pushes it into the RTSP factory's appsrc.
GstFlowReturn DeepStreamPipeline::on_new_sample(GstAppSink* appsink, gpointer data)
{
    auto* self = static_cast<DeepStreamPipeline*>(data);

    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_OK;

    g_mutex_lock(&self->rtsp_appsrc_mutex_);
    GstElement* appsrc = self->rtsp_appsrc_ ? GST_ELEMENT(gst_object_ref(self->rtsp_appsrc_)) : nullptr;
    g_mutex_unlock(&self->rtsp_appsrc_mutex_);

    if (appsrc) {
        gst_app_src_push_sample(GST_APP_SRC(appsrc), sample);
        gst_object_unref(appsrc);
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
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
    if (rtsp_appsrc_) {
        gst_object_unref(rtsp_appsrc_);
        rtsp_appsrc_ = nullptr;
    }
    g_mutex_unlock(&rtsp_appsrc_mutex_);
}
