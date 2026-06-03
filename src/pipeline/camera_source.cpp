#include "camera_source.h"
#include <sstream>
#include <stdexcept>

std::string build_source_description(const CameraConfig& cfg) {
    std::ostringstream ss;
    switch (cfg.type) {
        case CameraType::USB:
            ss << "v4l2src device=" << cfg.device
               << " ! image/jpeg"
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",framerate=" << cfg.fps << "/1";
            if (cfg.hw_decode) {
                // nvv4l2decoder uses Jetson NvJPEG hardware; Orin Nano has one unit —
                // only one camera can use this path. Outputs Y42B NVMM; nvvidconv→NV12.
                ss << " ! nvv4l2decoder mjpeg=1 ! nvvidconv";
            } else {
                // CPU path: jpegdec → nvvidconv (CPU→NVMM); ~30fps cap but no HW limit.
                ss << " ! jpegdec ! nvvidconv";
            }
            break;
        case CameraType::CSI:
            ss << "nvarguscamerasrc sensor-id=" << cfg.device
               << " ! video/x-raw(memory:NVMM)"
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",format=NV12"
               << ",framerate=" << cfg.fps << "/1";
            break;
        case CameraType::TEST:
            // Synthetic source for load/layout testing; pattern cycles by id.
            // is-live=true required because nvstreammux expects live sources.
            // nvvidconv converts CPU video/x-raw → NVMM NV12 for nvstreammux.
            ss << "videotestsrc is-live=true pattern=" << (cfg.id % 18)
               << " ! capsfilter caps=\"video/x-raw"
               << ",width="     << cfg.width
               << ",height="    << cfg.height
               << ",framerate=" << cfg.fps << "/1\""
               << " ! nvvidconv";
            break;
        default:
            throw std::invalid_argument("Unknown CameraType");
    }
    return ss.str();
}
