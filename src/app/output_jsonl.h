#pragma once
#include "app/detection_output.h"
#include <fstream>
#include <string>

// Writes each send_tracking_frame call as one JSON line — the replay diff artifact.
class JsonlOutput : public DetectionOutput {
public:
    explicit JsonlOutput(const std::string& path);

    void send_tracking_frame(const std::vector<TrackEvent>& events,
                             uint64_t timestamp_ns, bool healthy) override;

private:
    std::ofstream file_;
};
