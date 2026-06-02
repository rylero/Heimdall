#pragma once
#include <cstdint>

struct Detection {
    int      camera_id;
    int      class_id;
    float    confidence;
    float    left;        // pixels, 640×480 space
    float    top;         // pixels, 640×480 space
    float    width;       // pixels, 640×480 space
    float    height;      // pixels, 640×480 space
    uint64_t timestamp_ns;         // GStreamer buf_pts (pipeline running time, ns)
    uint64_t capture_monotonic_ns; // CLOCK_MONOTONIC absolute capture time (buf_pts + base_time)
};
