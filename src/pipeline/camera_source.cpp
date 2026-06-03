#include "camera_source.h"
#include <sstream>
#include <stdexcept>

std::string build_source_description(const CameraConfig& cfg) {
    std::ostringstream ss;
    switch (cfg.type) {
        case CameraType::USB:
            // nvv4l2decoder mjpeg=1 uses Jetson hardware JPEG decode and outputs NVMM directly,
            // avoiding the CPU jpegdec + nvvidconv path that caps throughput at ~30fps.
            // nvv4l2decoder outputs Y42B NVMM; nvvidconv converts to NV12 NVMM for nvstreammux.
            // Caps negotiation is left to nvvidconv↔nvstreammux; inline (memory:NVMM) caps
            // syntax breaks gst_parse_bin_from_description.
            ss << "v4l2src device=" << cfg.device
               << " ! image/jpeg"
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",framerate=" << cfg.fps << "/1"
               << " ! nvv4l2decoder mjpeg=1"
               << " ! nvvidconv";
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
            // Synthetic source for layout testing; pattern cycles through smpte/snow/colors by id.
            // is-live=true required because nvstreammux expects live sources.
            // capsfilter (quoted) avoids the parser treating "video/x-raw" as element "video" + URI.
            ss << "videotestsrc is-live=true pattern=" << (cfg.id % 18)
               << " ! capsfilter caps=\"video/x-raw"
               << ",width="     << cfg.width
               << ",height="    << cfg.height
               << ",framerate=" << cfg.fps << "/1\"";
            break;
        default:
            throw std::invalid_argument("Unknown CameraType");
    }
    return ss.str();
}
