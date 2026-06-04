#include "probe.h"
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <cstdio>
#include <atomic>
#include <chrono>

static std::atomic<int>      s_frame_count{0};
// Monotonic ns recorded when a buffer enters nvinfer (mux src probe).
// Safe with batch-size=1: only one frame in flight at a time.
static std::atomic<uint64_t> s_infer_entry_ns{0};

GstPadProbeReturn infer_entry_probe_cb(GstPad*, GstPadProbeInfo*, gpointer) {
    using clk = std::chrono::steady_clock;
    s_infer_entry_ns.store(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                clk::now().time_since_epoch()).count()),
        std::memory_order_relaxed);
    return GST_PAD_PROBE_OK;
}

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

    // Inference time: delta from mux-src probe to here (infer-src probe)
    using clk = std::chrono::steady_clock;
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clk::now().time_since_epoch()).count());
    const uint64_t entry_ns = s_infer_entry_ns.load(std::memory_order_relaxed);
    const float infer_ms = (entry_ns > 0) ? (now_ns - entry_ns) * 1e-6f : 0.f;

    static float s_infer_ms_avg = 0.f;
    if (infer_ms > 0.5f && infer_ms < 200.f)
        s_infer_ms_avg = s_infer_ms_avg * 0.9f + infer_ms * 0.1f;

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
            txt.display_text      = g_strdup_printf("FPS: %.1f  infer: %.1fms  CAM %d",
                                                     s_fps, s_infer_ms_avg, frame->source_id);
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
            std::fprintf(stderr, "[probe] cam%d frame=%d fps=%.1f infer_avg=%.1fms objs=%d\n",
                         frame->source_id, s_frame_count.load(), s_fps, s_infer_ms_avg, n);
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
