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
    uint64_t timestamp_ns;
};
