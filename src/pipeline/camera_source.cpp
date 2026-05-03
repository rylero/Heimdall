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
               << " ! jpegdec ! nvvidconv"
               << " ! video/x-raw,format=NV12";
            // Note: no explicit memory:NVMM feature here — nvstreammux's sink pad
            // template only accepts NVMM, so caps negotiation propagates that
            // requirement upstream to nvvidconv automatically.
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
