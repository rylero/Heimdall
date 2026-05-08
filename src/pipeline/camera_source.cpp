#include "camera_source.h"
#include <sstream>
#include <stdexcept>

std::string build_source_description(const CameraConfig& cfg) {
    std::ostringstream ss;
    switch (cfg.type) {
        case CameraType::USB:
            // Assumes camera outputs MJPEG (fastest path on Jetson).
            // If camera only supports raw YUY2, replace image/jpeg + jpegdec
            // with: video/x-raw,format=YUY2 ! videoconvert
            ss << "v4l2src device=" << cfg.device
               << " ! image/jpeg"
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",framerate=" << cfg.fps << "/1"
               << " ! jpegdec";
            // nvvidconv for NVMM conversion is added in pipeline.cpp for USB sources only;
            // CSI sources (nvarguscamerasrc) already output NVMM and link directly to mux.
            break;
        case CameraType::CSI:
            ss << "nvarguscamerasrc sensor-id=" << cfg.device
               << " ! video/x-raw(memory:NVMM)"
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",format=NV12"
               << ",framerate=" << cfg.fps << "/1";
            break;
        default:
            throw std::invalid_argument("Unknown CameraType");
    }
    return ss.str();
}
