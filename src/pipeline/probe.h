#pragma once
#include "detection.h"
#include <functional>
#include <gst/gst.h>
#include <vector>

using DetectionCallback = std::function<void(const std::vector<Detection>&)>;

GstPadProbeReturn detection_probe_cb(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);
