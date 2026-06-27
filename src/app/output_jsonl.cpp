#include "app/output_jsonl.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

static const char* event_type_str(TrackEventType t) {
    switch (t) {
        case TrackEventType::CONFIRMED: return "confirmed";
        case TrackEventType::UPDATED:   return "updated";
        case TrackEventType::LOST:      return "lost";
    }
    return "unknown";
}

JsonlOutput::JsonlOutput(const std::string& path)
    : file_(path, std::ios::out | std::ios::trunc)
{
    if (!file_)
        throw std::runtime_error("JsonlOutput: cannot open " + path);
}

void JsonlOutput::send_tracking_frame(const std::vector<TrackEvent>& events,
                                      uint64_t timestamp_ns, bool healthy) {
    json jevents = json::array();
    for (const auto& ev : events) {
        jevents.push_back({
            {"type",      event_type_str(ev.type)},
            {"track_id",  ev.object.track_id},
            {"class_id",  ev.object.class_id},
            {"x",         ev.object.x},
            {"y",         ev.object.y},
            {"vx",        ev.object.vx},
            {"vy",        ev.object.vy},
            {"ax",        ev.object.ax},
            {"ay",        ev.object.ay},
            {"conf",      ev.object.confidence},
        });
    }
    const json j = {
        {"ts_ns",   timestamp_ns},
        {"healthy", healthy},
        {"events",  std::move(jevents)},
    };
    file_ << j.dump() << '\n';
}
