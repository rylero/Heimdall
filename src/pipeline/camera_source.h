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
    // Clockwise rotation the pipeline applies before nvinfer, to correct for how the camera
    // is physically mounted. Only 0/90/180/270 are valid — a real camera can only be rotated,
    // never mirrored, so flips are deliberately not supported (they caused sign-convention bugs).
    int         rotation  = 0;
    // v4l2src source caps + buffer mode for USB cameras (5.21). pixel_format is the source
    // media type: "image/jpeg" (MJPEG, decoded) or a raw type like "video/x-raw,format=YUY2"
    // (no decoder). io_mode: 2=MMAP (default), 4=userptr, 1=RW — some cameras misbehave on MMAP.
    std::string pixel_format = "image/jpeg";
    int         io_mode      = 2;
};

std::string build_source_description(const CameraConfig& config);
