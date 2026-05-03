#include "pipeline.h"
#include <gst/app/gstappsink.h>
#include <stdexcept>
#include <string>

DeepStreamPipeline::DeepStreamPipeline(
    std::vector<CameraConfig> cameras,
    std::string               infer_config_path,
    DetectionCallback         on_detection,
    FrameCallback             on_frame
) : cameras_(std::move(cameras)),
    infer_config_path_(std::move(infer_config_path)),
    on_detection_(std::move(on_detection)),
    on_frame_(std::move(on_frame)) {}

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

    GstElement* sink = gst_element_factory_make("fakesink", "sink");
    if (!sink) throw std::runtime_error("Failed to create fakesink");
    g_object_set(sink, "sync", FALSE, nullptr);
    gst_bin_add(GST_BIN(pipeline_), sink);

    if (!gst_element_link(mux, infer))
        throw std::runtime_error("Failed to link mux to infer");

    if (on_frame_) {
        // Preview branch: infer → tee → [fakesink | nvvideoconvert → jpegenc → appsink]
        GstElement* tee     = gst_element_factory_make("tee",           "preview_tee");
        GstElement* q1      = gst_element_factory_make("queue",         "q_infer");
        GstElement* q2      = gst_element_factory_make("queue",         "q_preview");
        GstElement* conv    = gst_element_factory_make("nvvideoconvert","preview_conv");
        GstElement* enc     = gst_element_factory_make("jpegenc",       "preview_enc");
        GstElement* appsink = gst_element_factory_make("appsink",       "preview_appsink");
        if (!tee || !q1 || !q2 || !conv || !enc || !appsink)
            throw std::runtime_error("Failed to create preview pipeline elements");

        g_object_set(enc,     "quality", 60, nullptr);
        g_object_set(appsink, "emit-signals", TRUE, "sync", FALSE,
                               "max-buffers", 1, "drop", TRUE, nullptr);

        gst_bin_add_many(GST_BIN(pipeline_), tee, q1, q2, conv, enc, appsink, nullptr);

        if (!gst_element_link(infer, tee))
            throw std::runtime_error("Failed to link infer to tee");

        // Branch 1: tee → q1 → fakesink
        GstPad* tp1 = gst_element_get_request_pad(tee, "src_%u");
        GstPad* qs1 = gst_element_get_static_pad(q1, "sink");
        gst_pad_link(tp1, qs1);
        gst_object_unref(tp1); gst_object_unref(qs1);
        if (!gst_element_link(q1, sink))
            throw std::runtime_error("Failed to link q1 to fakesink");

        // Branch 2: tee → q2 → nvvideoconvert → jpegenc → appsink
        GstPad* tp2 = gst_element_get_request_pad(tee, "src_%u");
        GstPad* qs2 = gst_element_get_static_pad(q2, "sink");
        gst_pad_link(tp2, qs2);
        gst_object_unref(tp2); gst_object_unref(qs2);

        GstCaps* i420 = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "I420", nullptr);
        if (!gst_element_link(q2, conv) ||
            !gst_element_link_filtered(conv, enc, i420) ||
            !gst_element_link(enc, appsink))
            throw std::runtime_error("Failed to link preview branch");
        gst_caps_unref(i420);

        g_signal_connect(appsink, "new-sample", G_CALLBACK(appsink_cb), &on_frame_);
    } else {
        if (!gst_element_link(infer, sink))
            throw std::runtime_error("Failed to link infer to sink");
    }

    // Install pad probe on nvinfer src pad to extract detections each frame
    GstPad* infer_src = gst_element_get_static_pad(infer, "src");
    gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER,
        detection_probe_cb, &on_detection_, nullptr);
    gst_object_unref(infer_src);

    GstBus* bus = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus, bus_cb, this);
    gst_object_unref(bus);
}

GstFlowReturn DeepStreamPipeline::appsink_cb(GstElement* appsink, gpointer user_data) {
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    if (!sample) return GST_FLOW_OK;
    GstBuffer* buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        auto* cb = static_cast<FrameCallback*>(user_data);
        (*cb)(map.data, map.size);
        gst_buffer_unmap(buf, &map);
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
}
