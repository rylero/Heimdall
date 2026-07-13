#include "camera_source.h"
#include <sstream>
#include <stdexcept>

std::string build_source_description(const CameraConfig& cfg) {
    // nvvidconv flip-method (rotation only; no mirroring): 0=none, 1=ccw90, 2=180, 3=cw90.
    // cfg.rotation is clockwise degrees, so 90 (cw) -> 3 and 270 (cw == 90 ccw) -> 1.
    const int flip = (cfg.rotation == 180) ? 2
                   : (cfg.rotation == 90)  ? 3
                   : (cfg.rotation == 270) ? 1
                                           : 0;
    // Force NVMM output so the handoff to nvstreammux (which wants NVMM) is explicit rather
    // than left to implicit negotiation (5.21).
    const std::string vidconv = " ! nvvidconv flip-method=" + std::to_string(flip)
                              + " ! capsfilter caps=\"video/x-raw(memory:NVMM)\"";

    std::ostringstream ss;
    switch (cfg.type) {
        case CameraType::USB: {
            const bool is_jpeg = cfg.pixel_format.find("jpeg") != std::string::npos;
            ss << "v4l2src device=" << cfg.device
               << " io-mode=" << cfg.io_mode
               << " ! " << cfg.pixel_format
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",framerate=" << cfg.fps << "/1";
            if (is_jpeg) {
                ss << (cfg.hw_decode ? " ! nvv4l2decoder mjpeg=1" : " ! jpegparse ! jpegdec");
            }
            // Raw formats (e.g. YUY2) need no decoder — go straight to nvvidconv.
            ss << vidconv;
            break;
        }
        case CameraType::CSI:
            ss << "nvarguscamerasrc sensor-id=" << cfg.device
               << " ! video/x-raw(memory:NVMM)"
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",format=NV12"
               << ",framerate=" << cfg.fps << "/1"
               << vidconv;
            break;
        case CameraType::TEST:
            ss << "videotestsrc is-live=true pattern=" << (cfg.id % 18)
               << " ! capsfilter caps=\"video/x-raw"
               << ",width="     << cfg.width
               << ",height="    << cfg.height
               << ",framerate=" << cfg.fps << "/1\""
               << vidconv;
            break;
        default:
            throw std::invalid_argument("Unknown CameraType");
    }
    return ss.str();
}
