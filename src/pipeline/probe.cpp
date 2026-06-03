#include "probe.h"
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <cstdio>
#include <atomic>

static std::atomic<int> s_frame_count{0};

GstPadProbeReturn detection_probe_cb(GstPad* pad, GstPadProbeInfo* info, gpointer user_data) {
    auto* cb = static_cast<DetectionCallback*>(user_data);
    int n = ++s_frame_count;
    if (n <= 10 || n % 100 == 0)
        g_printerr("[probe] nvinfer frame %d\n", n);

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch) return GST_PAD_PROBE_OK;

    // Convert buf_pts (pipeline running time) → CLOCK_MONOTONIC absolute time.
    // base_time is set when the pipeline transitions to PLAYING and stays constant.
    GstElement* parent = gst_pad_get_parent_element(pad);
    const GstClockTime base_time = parent ? gst_element_get_base_time(parent) : 0;
    if (parent) gst_object_unref(parent);

    std::vector<Detection> detections;

    for (auto* lf = batch->frame_meta_list; lf; lf = lf->next) {
        auto* frame = static_cast<NvDsFrameMeta*>(lf->data);
        const uint64_t capture_ns = static_cast<uint64_t>(frame->buf_pts)
                                  + static_cast<uint64_t>(base_time);
        for (auto* lo = frame->obj_meta_list; lo; lo = lo->next) {
            auto* obj = static_cast<NvDsObjectMeta*>(lo->data);
            const auto& r = obj->rect_params;
            detections.push_back({
                .camera_id            = static_cast<int>(frame->source_id),
                .class_id             = static_cast<int>(obj->class_id),
                .confidence           = obj->confidence,
                .left                 = r.left,
                .top                  = r.top,
                .width                = r.width,
                .height               = r.height,
                .timestamp_ns         = frame->buf_pts,
                .capture_monotonic_ns = capture_ns,
            });
        }
    }

    (*cb)(detections);
    return GST_PAD_PROBE_OK;
}
