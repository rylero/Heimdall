#include "probe.h"
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#include <cstdio>
#include <atomic>
#include <chrono>

static std::atomic<int>      s_frame_count{0};
static std::atomic<uint64_t> s_infer_entry_ns{0};   // mux src → nvinfer entry
static std::atomic<uint64_t> s_decode_done_ns[8];   // per-camera: frame arrived at mux sink

uint64_t pipeline_frame_count() {
    return static_cast<uint64_t>(s_frame_count.load(std::memory_order_relaxed));
}

GstPadProbeReturn decode_done_probe_cb(GstPad*, GstPadProbeInfo*, gpointer user_data) {
    using clk = std::chrono::steady_clock;
    const int slot = *static_cast<int*>(user_data);
    if (slot >= 0 && slot < 8)
        s_decode_done_ns[slot].store(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    clk::now().time_since_epoch()).count()),
            std::memory_order_relaxed);
    return GST_PAD_PROBE_OK;
}

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

    // decode→mux time: averaged across all cameras in this batch
    static float s_decode_ms_avg = 0.f;
    {
        float sum = 0.f; int count = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t d = s_decode_done_ns[i].load(std::memory_order_relaxed);
            if (d > 0 && entry_ns > d && entry_ns - d < 100'000'000ULL) {
                sum += (entry_ns - d) * 1e-6f; ++count;
            }
        }
        if (count > 0) {
            float dm = sum / count;
            s_decode_ms_avg = s_decode_ms_avg * 0.9f + dm * 0.1f;
        }
    }

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch) return GST_PAD_PROBE_OK;

    // Capture-time stamp for this batch. buf_pts is GStreamer *running-time* (ns since the
    // pipeline hit PLAYING, ~0-based) — a different clock domain from PoseBuffer's keys, which
    // are steady_clock::now() (absolute CLOCK_MONOTONIC, ns-since-boot). Matching one against
    // the other made PoseBuffer::closest() always return the oldest pose (5.14). Stamp with
    // the probe's steady_clock time instead: the same domain as jetson_recv_ns, off true
    // capture only by the (small) decode+infer latency. now_ns is one instant for the whole
    // batch, which is correct — all cameras in a batch share a capture instant.
    const uint64_t capture_ns = now_ns;

    std::vector<Detection> detections;

    for (auto* lf = batch->frame_meta_list; lf; lf = lf->next) {
        auto* frame = static_cast<NvDsFrameMeta*>(lf->data);

        // FPS overlay — one label per frame, top-left corner
        NvDsDisplayMeta* dmeta = nvds_acquire_display_meta_from_pool(batch);
        if (dmeta) {
            dmeta->num_labels = 1;
            NvOSD_TextParams& txt = dmeta->text_params[0];
            txt.display_text      = g_strdup_printf("FPS: %.1f  dec: %.1fms  infer: %.1fms  CAM %d",
                                                     s_fps, s_decode_ms_avg, s_infer_ms_avg, frame->source_id);
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
            std::fprintf(stderr, "[probe] cam%d frame=%d fps=%.1f dec_avg=%.1fms infer_avg=%.1fms total=%.1fms objs=%d\n",
                         frame->source_id, s_frame_count.load(), s_fps,
                         s_decode_ms_avg, s_infer_ms_avg, s_decode_ms_avg + s_infer_ms_avg, n);
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
                .timestamp_ns         = capture_ns,   // steady_clock domain (was buf_pts) — see above
                .capture_monotonic_ns = capture_ns,
            });
        }
    }

    (*cb)(detections);
    return GST_PAD_PROBE_OK;
}
