#pragma once
#include "tracker/track_event.h"
#include <cstdint>
#include <vector>

struct DetectionOutput {
    virtual void send_tracking_frame(const std::vector<TrackEvent>& events,
                                     uint64_t timestamp_ns, bool healthy) = 0;
    virtual ~DetectionOutput() = default;
};
