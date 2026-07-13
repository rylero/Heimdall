#include "pipeline.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

// Stable per-slot indices passed as user_data to decode_done_probe_cb.
static int s_cam_slots[8] = {0, 1, 2, 3, 4, 5, 6, 7};

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
        "batched-push-timeout", 1000000,
        "live-source",          TRUE,
        nullptr);
    gst_bin_add(GST_BIN(pipeline_), mux);

    // camera_id → tee element, populated for cameras that have at least one mirror
    std::unordered_map<int, GstElement*> tees;

    // Insert a leaky queue between a tee output pad and the mux sink.
    // leaky=2 (downstream) drops the oldest buffer when full so neither tee branch
    // can stall the other — preventing backpressure from reaching v4l2src.
    auto link_via_leaky_queue = [&](GstPad* tee_src_pad, int slot) {
        std::string qname = "tee_q_" + std::to_string(slot);
        GstElement* q = gst_element_factory_make("queue", qname.c_str());
        if (!q) throw std::runtime_error("Failed to create leaky queue for slot " + std::to_string(slot));
        g_object_set(q,
            "max-size-buffers", 2u,
            "max-size-bytes",   0u,
            "max-size-time",    guint64(0),
            "leaky",            2u,  // GST_QUEUE_LEAK_DOWNSTREAM
            nullptr);
        gst_bin_add(GST_BIN(pipeline_), q);

        GstPad* q_sink = gst_element_get_static_pad(q, "sink");
        GstPad* q_src  = gst_element_get_static_pad(q, "src");
        GstPad* mux_sink = gst_element_get_request_pad(mux, ("sink_" + std::to_string(slot)).c_str());
        if (!q_sink || !q_src || !mux_sink)
            throw std::runtime_error("Failed to get pads for leaky queue slot " + std::to_string(slot));
        if (gst_pad_link(tee_src_pad, q_sink) != GST_PAD_LINK_OK)
            throw std::runtime_error("Failed to link tee → queue for slot " + std::to_string(slot));
        if (gst_pad_link(q_src, mux_sink) != GST_PAD_LINK_OK)
            throw std::runtime_error("Failed to link queue → mux for slot " + std::to_string(slot));
        gst_object_unref(q_sink);
        gst_object_unref(q_src);
        gst_object_unref(mux_sink);
    };

    // First pass: build source bins for real cameras; insert tee if any camera mirrors them
    for (int i = 0; i < static_cast<int>(cameras_.size()); ++i) {
        if (cameras_[i].mirror_of >= 0) continue;

        bool needs_tee = false;
        for (auto& c : cameras_)
            if (c.mirror_of == cameras_[i].id) { needs_tee = true; break; }

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

        if (needs_tee) {
            GstElement* tee = gst_element_factory_make("tee", ("tee_" + std::to_string(cameras_[i].id)).c_str());
            if (!tee) throw std::runtime_error("Failed to create tee for camera " + std::to_string(i));
            gst_bin_add(GST_BIN(pipeline_), tee);

            GstPad* src_ghost = gst_element_get_static_pad(src, "src");
            GstPad* tee_sink  = gst_element_get_static_pad(tee, "sink");
            if (gst_pad_link(src_ghost, tee_sink) != GST_PAD_LINK_OK)
                throw std::runtime_error("Failed to link src to tee for camera " + std::to_string(i));
            gst_object_unref(src_ghost);
            gst_object_unref(tee_sink);

            tees[cameras_[i].id] = tee;
            GstPad* tee_src = gst_element_get_request_pad(tee, "src_%u");
            if (!tee_src) throw std::runtime_error("No tee src pad for camera " + std::to_string(i));
            link_via_leaky_queue(tee_src, i);
            gst_object_unref(tee_src);
        } else {
            GstPad* out_pad  = gst_element_get_static_pad(src, "src");
            GstPad* mux_sink = gst_element_get_request_pad(mux, ("sink_" + std::to_string(i)).c_str());
            if (!out_pad || !mux_sink)
                throw std::runtime_error("No output pad for camera " + std::to_string(i));
            if (gst_pad_link(out_pad, mux_sink) != GST_PAD_LINK_OK)
                throw std::runtime_error("Failed to link src to mux for camera " + std::to_string(i));
            gst_object_unref(out_pad);
            gst_object_unref(mux_sink);
        }
    }

    // Second pass: link mirror cameras from their source's tee via leaky queues
    for (int i = 0; i < static_cast<int>(cameras_.size()); ++i) {
        if (cameras_[i].mirror_of < 0) continue;

        auto it = tees.find(cameras_[i].mirror_of);
        if (it == tees.end())
            throw std::runtime_error("Mirror camera " + std::to_string(i) +
                " references camera " + std::to_string(cameras_[i].mirror_of) +
                " which has no tee — check mirror_of ids");

        GstPad* tee_src  = gst_element_get_request_pad(it->second, "src_%u");
        if (!tee_src)
            throw std::runtime_error("Failed to get tee src pad for mirror camera " + std::to_string(i));
        link_via_leaky_queue(tee_src, i);
        gst_object_unref(tee_src);
    }

    GstElement* infer = gst_element_factory_make("nvinfer", "infer");
    if (!infer) throw std::runtime_error("Failed to create nvinfer");
    g_object_set(infer, "config-file-path", infer_config_path_.c_str(), nullptr);
    gst_bin_add(GST_BIN(pipeline_), infer);

    // Decode-done probes: one per mux sink pad, fires when a decoded frame arrives.
    for (int i = 0; i < static_cast<int>(cameras_.size()); ++i) {
        std::string pad_name = "sink_" + std::to_string(i);
        GstPad* sink = gst_element_get_static_pad(mux, pad_name.c_str());
        if (sink) {
            gst_pad_add_probe(sink, GST_PAD_PROBE_TYPE_BUFFER,
                decode_done_probe_cb, &s_cam_slots[i], nullptr);
            gst_object_unref(sink);
        }
    }

    // Entry probe on mux src: timestamps the moment a buffer enters nvinfer.
    GstPad* mux_src = gst_element_get_static_pad(mux, "src");
    gst_pad_add_probe(mux_src, GST_PAD_PROBE_TYPE_BUFFER,
        infer_entry_probe_cb, nullptr, nullptr);
    gst_object_unref(mux_src);

    // Exit probe on infer src: fires after inference, computes inference time.
    // Also before tiling so frame->source_id still maps to original camera index.
    GstPad* infer_src = gst_element_get_static_pad(infer, "src");
    gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER,
        detection_probe_cb, &on_detection_, nullptr);
    gst_object_unref(infer_src);

    // queue decouples nvstreammux's streaming thread from the encoding chain.
    // leaky=2 (downstream) is essential (5.15): the detection probe sits on infer_src,
    // *upstream* of this queue, so if the encode/RTMP chain stalls (MediaMTX restart,
    // network hiccup) a non-leaky queue would backpressure through nvinfer and stop
    // inference — i.e. stop detections. Leaking here lets the display branch drop frames
    // instead of ever stalling the detection path.
    GstElement* queue_post_infer = gst_element_factory_make("queue", "queue_post_infer");
    if (!queue_post_infer) throw std::runtime_error("Failed to create queue");
    g_object_set(queue_post_infer,
        "max-size-buffers", 4u,
        "max-size-bytes",   0u,
        "max-size-time",    guint64(0),
        "leaky",            2u,  // GST_QUEUE_LEAK_DOWNSTREAM
        nullptr);
    gst_bin_add(GST_BIN(pipeline_), queue_post_infer);

    // Tile all camera streams into a roughly square grid.
    const int tiler_cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(cameras_.size()))));
    const int tiler_rows = static_cast<int>(std::ceil(static_cast<double>(cameras_.size()) / tiler_cols));
    GstElement* tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
    if (!tiler) throw std::runtime_error("Failed to create nvmultistreamtiler");
    g_object_set(tiler,
        "rows",    static_cast<guint>(tiler_rows),
        "columns", static_cast<guint>(tiler_cols),
        "width",   static_cast<guint>(cameras_[0].width  * tiler_cols),
        "height",  static_cast<guint>(cameras_[0].height * tiler_rows),
        nullptr);
    gst_bin_add(GST_BIN(pipeline_), tiler);

    GstElement* osd = gst_element_factory_make("nvdsosd", "osd");
    if (!osd) throw std::runtime_error("Failed to create nvdsosd");
    g_object_set(osd, "process-mode", 0, nullptr);
    gst_bin_add(GST_BIN(pipeline_), osd);

    GstElement* conv_out = gst_element_factory_make("nvvidconv", "conv_out");
    if (!conv_out) throw std::runtime_error("Failed to create nvvidconv");
    gst_bin_add(GST_BIN(pipeline_), conv_out);

    GstElement* test_enc = gst_element_factory_make("nvv4l2h264enc", nullptr);
    const bool  hw_enc   = (test_enc != nullptr);
    if (test_enc) gst_object_unref(test_enc);
    else g_printerr("[pipeline] nvv4l2h264enc unavailable, using x264enc\n");

    GstElement* encoder;
    GstCaps*    enc_caps;
    if (hw_enc) {
        encoder  = gst_element_factory_make("nvv4l2h264enc", "encoder");
        if (!encoder) throw std::runtime_error("Failed to create nvv4l2h264enc");
        g_object_set(encoder,
            "bitrate",     static_cast<guint>(4000000),
            "idrinterval", static_cast<guint>(30),
            nullptr);
        enc_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
    } else {
        encoder  = gst_element_factory_make("x264enc", "encoder");
        if (!encoder) throw std::runtime_error("No H264 encoder available");
        g_object_set(encoder,
            "bitrate",      4000u,
            "tune",         0x4u,  // zerolatency
            "speed-preset", 1u,    // ultrafast
            nullptr);
        enc_caps = gst_caps_from_string("video/x-raw,format=I420");
    }
    gst_bin_add(GST_BIN(pipeline_), encoder);

    GstElement* caps_out = gst_element_factory_make("capsfilter", "caps_out");
    if (!caps_out) throw std::runtime_error("Failed to create capsfilter");
    g_object_set(caps_out, "caps", enc_caps, nullptr);
    gst_caps_unref(enc_caps);
    gst_bin_add(GST_BIN(pipeline_), caps_out);

    GstElement* flvmux = gst_element_factory_make("flvmux", "flvmux");
    if (!flvmux) throw std::runtime_error("Failed to create flvmux");
    g_object_set(flvmux, "streamable", TRUE, nullptr);
    gst_bin_add(GST_BIN(pipeline_), flvmux);

    GstElement* rtmp_sink = gst_element_factory_make("rtmpsink", "rtmp_sink");
    if (!rtmp_sink) throw std::runtime_error("Failed to create rtmpsink");
    g_object_set(rtmp_sink, "location", "rtmp://127.0.0.1:1935/live/ds-test", nullptr);
    gst_bin_add(GST_BIN(pipeline_), rtmp_sink);

    if (!gst_element_link(mux,              infer))            throw std::runtime_error("Failed to link mux→infer");
    if (!gst_element_link(infer,            queue_post_infer)) throw std::runtime_error("Failed to link infer→queue");
    if (!gst_element_link(queue_post_infer, tiler))            throw std::runtime_error("Failed to link queue→tiler");
    if (!gst_element_link(tiler,            osd))              throw std::runtime_error("Failed to link tiler→osd");
    if (!gst_element_link(osd,              conv_out))         throw std::runtime_error("Failed to link osd→conv_out");
    if (!gst_element_link(conv_out,         caps_out))         throw std::runtime_error("Failed to link conv_out→caps_out");
    if (!gst_element_link(caps_out,         encoder))          throw std::runtime_error("Failed to link caps_out→encoder");
    if (!gst_element_link(encoder,          flvmux))           throw std::runtime_error("Failed to link encoder→flvmux");
    if (!gst_element_link(flvmux,           rtmp_sink))        throw std::runtime_error("Failed to link flvmux→rtmp_sink");

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline_), GST_DEBUG_GRAPH_SHOW_ALL, "heimdall-pipeline");

    GstBus* bus = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus, bus_cb, this);
    gst_object_unref(bus);

    gst_pipeline_set_latency(GST_PIPELINE(pipeline_), 500 * GST_MSECOND);

    std::printf("RTSP stream: rtsp://0.0.0.0:%d/live/ds-test  (MediaMTX ingests RTMP on :1935)\n", RTSP_SERV_PORT);
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

void DeepStreamPipeline::restart() {
    if (!pipeline_) return;
    g_printerr("[pipeline] stall detected — cycling NULL→PLAYING to recover\n");
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
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
