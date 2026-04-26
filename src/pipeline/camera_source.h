#pragma once
#include <string>

enum class CameraType { USB, CSI };

struct CameraConfig {
    int         id;
    CameraType  type;
    std::string device;     // "/dev/video0" for USB, "0" for CSI sensor index
    int         width  = 640;
    int         height = 480;
    int         fps    = 60; // CSI sensors (IMX219/IMX477) may cap at 30 fps at 640×480 — verify sensor mode
};

std::string build_source_description(const CameraConfig& config);
