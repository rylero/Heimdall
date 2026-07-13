#pragma once
#include "comm/comm_layer.h"
#include "pipeline/detection.h"
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// Thread-safe appender — called from pose-recv thread and det-worker thread concurrently.
class RecordingWriter {
public:
    explicit RecordingWriter(const std::string& path);

    void write_pose(const CommLayer::TimestampedPose& p);
    void write_frame(const std::vector<Detection>& dets);

private:
    std::mutex    mu_;
    std::ofstream file_;
};

// Parses a recording produced by RecordingWriter into ordered poses + frames.
struct RecordedFrame {
    std::vector<Detection> dets;               // may be empty (heartbeat frame)
    uint64_t               frame_capture_ns = 0; // frame-level capture time, incl. empty frames (5.17)
};

struct RecordingData {
    std::vector<CommLayer::TimestampedPose> poses;
    std::vector<RecordedFrame>              frames;
};

RecordingData load_recording(const std::string& path);
