#pragma once
#include <string>

enum class CameraType { USB, CSI, TEST };

struct CameraConfig {
    int         id;
    CameraType  type;
    std::string device;     // "/dev/video0" for USB, "0" for CSI sensor index
    int         width     = 640;
    int         height    = 480;
    int         fps       = 60;
    int         mirror_of = -1; // >=0: mirror that camera id via tee (no source bin created)
    bool        hw_decode = true; // false: use CPU jpegdec (Orin Nano has 1 NvJPEG unit)
};

std::string build_source_description(const CameraConfig& config);
