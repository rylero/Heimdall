// Integration tests for the post-DeepStream encoder chain.
// Uses videotestsrc to drive the pipeline — no real camera or nvinfer needed.
// Requires GStreamer (built inside Docker); skips gracefully when elements are absent.
//
// Run via: ctest -R pipeline_chain --output-on-failure
//
// HOW TO INTERPRET FAILURES:
//   encoder_chain_flows  → x264enc/nvvidconv path broken; caps negotiation issue
//   nvvidconv_nv12_to_i420 → nvvidconv can't convert NV12→I420; format unsupported
//   osd_chain_flows      → nvdsosd CPU mode breaks the frame flow; switch process-mode

#include <catch2/catch_test_macros.hpp>
#include <gst/gst.h>
#include <atomic>

static GstPadProbeReturn count_buffers(GstPad*, GstPadProbeInfo*, gpointer data) {
    ++(*static_cast<std::atomic<int>*>(data));
    return GST_PAD_PROBE_OK;
}

static void attach_sink_probe(GstElement* element, std::atomic<int>& counter) {
    GstPad* pad = gst_element_get_static_pad(element, "sink");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, count_buffers, &counter, nullptr);
    gst_object_unref(pad);
}

// Run a pipeline until EOS or error; return error message if error, empty string on EOS.
static std::string run_to_eos(GstElement* pipeline, int timeout_sec = 8) {
    auto ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE)
        return "state change to PLAYING failed";

    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* msg = gst_bus_timed_pop_filtered(
        bus,
        timeout_sec * GST_SECOND,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    gst_object_unref(bus);

    std::string err;
    if (!msg) {
        err = "timed out after " + std::to_string(timeout_sec) + "s — pipeline stalled";
    } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* e; gchar* dbg;
        gst_message_parse_error(msg, &e, &dbg);
        err = std::string(e->message) + (dbg ? std::string("\n") + dbg : "");
        g_error_free(e); g_free(dbg);
    }
    if (msg) gst_message_unref(msg);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return err;
}

// ---------------------------------------------------------------------------
// Test 1: basic encoder chain
//   videotestsrc → videoconvert → video/x-raw,I420 → x264enc → rtph264pay → udpsink
//
// This is the simplest possible version of our post-infer path.  If this fails,
// x264enc or rtph264pay itself is broken in the container.
// ---------------------------------------------------------------------------
TEST_CASE("encoder_chain_flows: x264enc produces RTP packets", "[pipeline][chain]") {
    gst_init(nullptr, nullptr);

    GstElement* x264 = gst_element_factory_make("x264enc", nullptr);
    if (!x264) SKIP("x264enc not available in this build");
    gst_object_unref(x264);

    std::atomic<int> rtp_count{0};

    GstElement* pipeline = gst_pipeline_new("enc-chain-test");
    GstElement* src      = gst_element_factory_make("videotestsrc",  "src");
    GstElement* conv     = gst_element_factory_make("videoconvert",  "conv");
    GstElement* cf       = gst_element_factory_make("capsfilter",    "cf");
    GstElement* encoder  = gst_element_factory_make("x264enc",       "enc");
    GstElement* pay      = gst_element_factory_make("rtph264pay",    "pay");
    GstElement* sink     = gst_element_factory_make("fakesink",      "sink");
    REQUIRE((src && conv && cf && encoder && pay && sink));

    g_object_set(src, "num-buffers", 60, nullptr);
    GstCaps* caps = gst_caps_from_string("video/x-raw,format=I420,width=640,height=480,framerate=30/1");
    g_object_set(cf, "caps", caps, nullptr);
    gst_caps_unref(caps);
    g_object_set(pay, "config-interval", 1, "pt", 96, nullptr);
    g_object_set(sink, "sync", FALSE, nullptr);

    attach_sink_probe(sink, rtp_count);

    gst_bin_add_many(GST_BIN(pipeline), src, conv, cf, encoder, pay, sink, nullptr);
    REQUIRE(gst_element_link_many(src, conv, cf, encoder, pay, sink, nullptr));

    auto err = run_to_eos(pipeline);
    INFO("RTP frames at fakesink: " << rtp_count.load());
    REQUIRE(err.empty());
    REQUIRE(rtp_count.load() > 0);
}

// ---------------------------------------------------------------------------
// Test 2: nvvidconv NV12→I420 conversion (system memory path)
//   videotestsrc NV12 → nvvidconv → video/x-raw,I420 → fakesink
//
// This exercises the exact conversion that conv_out must do when feeding x264enc.
// nvvidconv should be able to handle system-memory NV12→I420 without NVMM.
// If this fails, the caps negotiation between nvvidconv and the encoder is broken.
// ---------------------------------------------------------------------------
TEST_CASE("nvvidconv_nv12_to_i420: format conversion works", "[pipeline][chain]") {
    gst_init(nullptr, nullptr);

    GstElement* probe_elem = gst_element_factory_make("nvvidconv", nullptr);
    if (!probe_elem) SKIP("nvvidconv not available");
    gst_object_unref(probe_elem);

    std::atomic<int> frame_count{0};

    GstElement* pipeline = gst_pipeline_new("conv-test");
    GstElement* src      = gst_element_factory_make("videotestsrc", "src");
    GstElement* src_cf   = gst_element_factory_make("capsfilter",   "src_cf");
    GstElement* conv     = gst_element_factory_make("nvvidconv",    "conv");
    GstElement* out_cf   = gst_element_factory_make("capsfilter",   "out_cf");
    GstElement* sink     = gst_element_factory_make("fakesink",     "sink");
    REQUIRE((src && src_cf && conv && out_cf && sink));

    g_object_set(src, "num-buffers", 30, nullptr);

    GstCaps* nv12 = gst_caps_from_string("video/x-raw,format=NV12,width=640,height=480,framerate=30/1");
    g_object_set(src_cf, "caps", nv12, nullptr);
    gst_caps_unref(nv12);

    GstCaps* i420 = gst_caps_from_string("video/x-raw,format=I420");
    g_object_set(out_cf, "caps", i420, nullptr);
    gst_caps_unref(i420);

    g_object_set(sink, "sync", FALSE, nullptr);
    attach_sink_probe(sink, frame_count);

    gst_bin_add_many(GST_BIN(pipeline), src, src_cf, conv, out_cf, sink, nullptr);
    REQUIRE(gst_element_link_many(src, src_cf, conv, out_cf, sink, nullptr));

    auto err = run_to_eos(pipeline);
    INFO("Converted frames: " << frame_count.load());
    REQUIRE(err.empty());
    REQUIRE(frame_count.load() > 0);
}

// ---------------------------------------------------------------------------
// Test 3: nvdsosd (CPU mode) does not stall frame flow
//   videotestsrc NV12 → nvdsosd(process-mode=1) → nvvidconv → I420 → x264enc → fakesink
//
// This replicates the exact post-infer chain in pipeline.cpp.
// If THIS test fails but encoder_chain_flows passes, nvdsosd CPU mode is the culprit.
// In that case, try process-mode=0 (GPU/CUDA — works fine headless on x86 with CUDA).
// ---------------------------------------------------------------------------
TEST_CASE("osd_chain_flows: nvdsosd CPU mode does not stall pipeline", "[pipeline][chain]") {
    gst_init(nullptr, nullptr);

    GstElement* osd_probe = gst_element_factory_make("nvdsosd", nullptr);
    if (!osd_probe) SKIP("nvdsosd not available (non-DeepStream environment)");
    gst_object_unref(osd_probe);

    GstElement* enc_probe = gst_element_factory_make("x264enc", nullptr);
    if (!enc_probe) SKIP("x264enc not available");
    gst_object_unref(enc_probe);

    std::atomic<int> enc_frame_count{0};

    GstElement* pipeline = gst_pipeline_new("osd-chain-test");
    GstElement* src      = gst_element_factory_make("videotestsrc", "src");
    GstElement* src_cf   = gst_element_factory_make("capsfilter",   "src_cf");
    GstElement* osd      = gst_element_factory_make("nvdsosd",      "osd");
    GstElement* conv     = gst_element_factory_make("nvvidconv",    "conv");
    if (!conv) conv = gst_element_factory_make("videoconvert", "conv");
    GstElement* out_cf   = gst_element_factory_make("capsfilter",   "out_cf");
    GstElement* encoder  = gst_element_factory_make("x264enc",      "enc");
    GstElement* sink     = gst_element_factory_make("fakesink",     "sink");
    REQUIRE((src && src_cf && osd && conv && out_cf && encoder && sink));

    g_object_set(src, "num-buffers", 30, nullptr);

    // nvdsosd requires NV12 input; feed it the same format the tiler produces
    GstCaps* nv12 = gst_caps_from_string("video/x-raw,format=NV12,width=640,height=480,framerate=30/1");
    g_object_set(src_cf, "caps", nv12, nullptr);
    gst_caps_unref(nv12);

    // Same process-mode used in production pipeline
    g_object_set(osd, "process-mode", 1, nullptr);

    GstCaps* i420 = gst_caps_from_string("video/x-raw,format=I420");
    g_object_set(out_cf, "caps", i420, nullptr);
    gst_caps_unref(i420);

    g_object_set(sink, "sync", FALSE, nullptr);
    attach_sink_probe(sink, enc_frame_count);

    gst_bin_add_many(GST_BIN(pipeline), src, src_cf, osd, conv, out_cf, encoder, sink, nullptr);
    REQUIRE(gst_element_link_many(src, src_cf, osd, conv, out_cf, encoder, sink, nullptr));

    auto err = run_to_eos(pipeline);
    INFO("Frames past encoder: " << enc_frame_count.load());

    // Report the actual error clearly to help diagnose process-mode issues
    if (!err.empty())
        FAIL("osd→conv→encoder chain error (try process-mode=0/CPU in pipeline.cpp): " << err);

    REQUIRE(enc_frame_count.load() > 0);
}

// ---------------------------------------------------------------------------
// Test 4: NVMM→system memory path (nvdsosd NVMM output → nvvidconv → system I420)
//   Only meaningful inside the DeepStream Docker container with CUDA.
//   Simulates the NVMM memory path that the real pipeline uses.
// ---------------------------------------------------------------------------
TEST_CASE("nvmm_to_system_conv: NVMM NV12 flows through nvvidconv to system I420", "[pipeline][chain][nvmm]") {
    gst_init(nullptr, nullptr);

    GstElement* conv_probe = gst_element_factory_make("nvvidconv", nullptr);
    if (!conv_probe) SKIP("nvvidconv not available");
    gst_object_unref(conv_probe);

    std::atomic<int> frame_count{0};

    GstElement* pipeline = gst_pipeline_new("nvmm-conv-test");
    GstElement* src      = gst_element_factory_make("videotestsrc",  "src");
    GstElement* src_cf   = gst_element_factory_make("capsfilter",    "src_cf");
    GstElement* upload   = gst_element_factory_make("nvvidconv",     "upload");
    GstElement* nvmm_cf  = gst_element_factory_make("capsfilter",    "nvmm_cf");
    GstElement* conv     = gst_element_factory_make("nvvidconv",     "conv");
    GstElement* out_cf   = gst_element_factory_make("capsfilter",    "out_cf");
    GstElement* sink     = gst_element_factory_make("fakesink",      "sink");
    REQUIRE((src && src_cf && upload && nvmm_cf && conv && out_cf && sink));

    g_object_set(src, "num-buffers", 30, nullptr);

    GstCaps* nv12_sys = gst_caps_from_string("video/x-raw,format=NV12,width=640,height=480,framerate=30/1");
    g_object_set(src_cf, "caps", nv12_sys, nullptr);
    gst_caps_unref(nv12_sys);

    // Force intermediate NVMM memory (simulates nvdsosd output in real pipeline)
    GstCaps* nv12_nvmm = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12,width=640,height=480");
    g_object_set(nvmm_cf, "caps", nv12_nvmm, nullptr);
    gst_caps_unref(nv12_nvmm);

    // conv_out must produce system I420 for x264enc (same as production pipeline)
    GstCaps* i420_sys = gst_caps_from_string("video/x-raw,format=I420");
    g_object_set(out_cf, "caps", i420_sys, nullptr);
    gst_caps_unref(i420_sys);

    g_object_set(sink, "sync", FALSE, nullptr);
    attach_sink_probe(sink, frame_count);

    gst_bin_add_many(GST_BIN(pipeline), src, src_cf, upload, nvmm_cf, conv, out_cf, sink, nullptr);
    REQUIRE(gst_element_link_many(src, src_cf, upload, nvmm_cf, conv, out_cf, sink, nullptr));

    auto err = run_to_eos(pipeline);
    INFO("Frames through NVMM→system conversion: " << frame_count.load());

    if (!err.empty())
        FAIL("NVMM→system caps negotiation failed (nvvidconv can't convert NVMM NV12→system I420): " << err);

    REQUIRE(frame_count.load() > 0);
}
