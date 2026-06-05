#pragma once
#include "detection.h"
#include <functional>
#include <gst/gst.h>
#include <vector>

using DetectionCallback = std::function<void(const std::vector<Detection>&)>;

GstPadProbeReturn detection_probe_cb(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);
// Attach to mux src pad to timestamp the start of inference
GstPadProbeReturn infer_entry_probe_cb(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);
// Attach to each mux sink pad; user_data = int* pointing to the camera slot index (0..N-1)
GstPadProbeReturn decode_done_probe_cb(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);
