#include "camera_source.h"
#include <sstream>
#include <stdexcept>

std::string build_source_description(const CameraConfig& cfg) {
    // nvvidconv flip-method: 0=none, 4=horiz, 6=vert, 2=both(180°)
    const int flip = (cfg.flip_h && cfg.flip_v) ? 2
                   : cfg.flip_h                 ? 4
                   : cfg.flip_v                 ? 6
                                                : 0;
    const std::string vidconv = " ! nvvidconv flip-method=" + std::to_string(flip);

    std::ostringstream ss;
    switch (cfg.type) {
        case CameraType::USB:
            ss << "v4l2src device=" << cfg.device
               << " ! image/jpeg"
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",framerate=" << cfg.fps << "/1";
            if (cfg.hw_decode) {
                ss << " ! nvv4l2decoder mjpeg=1" << vidconv;
            } else {
                ss << " ! jpegdec" << vidconv;
            }
            break;
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
