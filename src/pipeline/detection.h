#pragma once
#include <cstdint>

/*
    Basic structure for a detection in an Deepstream object detection popeline
    Note that all pixel measurements are assumed to be in a 640x480 space
*/
struct Detection {
    int   camera_id;
    int   class_id;
    float confidence;
    
    // pixel measurements
    float left;
    float top;
    float width;
    float height;

    // timestamps for detection
    uint64_t timestamp_ns;         // GStreamer buf_pts (pipeline running time, ns)
    uint64_t capture_monotonic_ns; // CLOCK_MONOTONIC absolute capture time (buf_pts + base_time)
};
