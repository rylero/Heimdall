#include "probe.h"
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <cstdio>
#include <atomic>
#include <chrono>

static std::atomic<int> s_frame_count{0};

GstPadProbeReturn detection_probe_cb(GstPad* pad, GstPadProbeInfo* info, gpointer user_data) {
    auto* cb = static_cast<DetectionCallback*>(user_data);
    ++s_frame_count;

    // FPS: exponential moving average over the probe's firing cadence
    static auto  s_last_ts = std::chrono::steady_clock::now();
    static float s_fps     = 0.f;
    auto  now = std::chrono::steady_clock::now();
    float dt  = std::chrono::duration<float>(now - s_last_ts).count();
    s_last_ts = now;
    if (dt > 0.005f && dt < 1.f)   // 5ms minimum avoids microsecond artifact on first call
        s_fps = s_fps * 0.85f + (1.f / dt) * 0.15f;

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch) return GST_PAD_PROBE_OK;

    const GstClockTime base_time = 0;

    std::vector<Detection> detections;

    for (auto* lf = batch->frame_meta_list; lf; lf = lf->next) {
        auto* frame = static_cast<NvDsFrameMeta*>(lf->data);
        const uint64_t capture_ns = static_cast<uint64_t>(frame->buf_pts)
                                  + static_cast<uint64_t>(base_time);

        // FPS overlay — one label per frame, top-left corner
        NvDsDisplayMeta* dmeta = nvds_acquire_display_meta_from_pool(batch);
        if (dmeta) {
            dmeta->num_labels = 1;
            NvOSD_TextParams& txt = dmeta->text_params[0];
            txt.display_text      = g_strdup_printf("FPS: %.1f  CAM %d",
                                                     s_fps, frame->source_id);
            txt.x_offset          = 10;
            txt.y_offset          = 10;
            txt.font_params.font_name  = const_cast<gchar*>("Serif Bold");
            txt.font_params.font_size  = 18;
            txt.font_params.font_color = {1.f, 1.f, 1.f, 1.f};
            txt.set_bg_clr             = 1;
            txt.text_bg_clr            = {0.f, 0.f, 0.f, 0.5f};
            nvds_add_display_meta_to_frame(frame, dmeta);
        }

        // Debug: log object count every 100 frames to confirm nvinfer attaches metadata
        if (s_frame_count % 100 == 0) {
            int n = 0; for (auto* l = frame->obj_meta_list; l; l = l->next) ++n;
            std::fprintf(stderr, "[probe] cam%d frame=%d obj_meta count=%d\n",
                         frame->source_id, s_frame_count.load(), n);
        }

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
