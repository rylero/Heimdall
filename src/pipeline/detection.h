#pragma once
#include <cstdint>

struct Detection {
    int      camera_id;
    int      class_id;
    float    confidence;
    float    left;
    float    top;
    float    width;
    float    height;
    uint64_t timestamp_ns;
};
