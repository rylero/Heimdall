#include "probe.h"
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

GstPadProbeReturn detection_probe_cb(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
    auto* cb = static_cast<DetectionCallback*>(user_data);

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch) return GST_PAD_PROBE_OK;

    std::vector<Detection> detections;

    for (auto* lf = batch->frame_meta_list; lf; lf = lf->next) {
        auto* frame = static_cast<NvDsFrameMeta*>(lf->data);
        for (auto* lo = frame->obj_meta_list; lo; lo = lo->next) {
            auto* obj = static_cast<NvDsObjectMeta*>(lo->data);
            const auto& r = obj->rect_params;
            detections.push_back({
                .camera_id    = static_cast<int>(frame->source_id),
                .class_id     = static_cast<int>(obj->class_id),
                .confidence   = obj->confidence,
                .left         = r.left,
                .top          = r.top,
                .width        = r.width,
                .height       = r.height,
                .timestamp_ns = frame->buf_pts,
            });
        }
    }

    (*cb)(detections);
    return GST_PAD_PROBE_OK;
}
