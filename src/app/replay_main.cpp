#include "app/detection_processor.h"
#include "app/output_jsonl.h"
#include "app/recording.h"
#include "config/camera_config_loader.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s <recording.jsonl> <camera_config_dir> <output.jsonl> [options]\n"
        "  --bypass-tracker   pass raw detections through without JPDAF\n"
        "  --log-tracking     write tracker_log.csv alongside output\n",
        argv0);
}

int main(int argc, char** argv) {
    if (argc < 4) { usage(argv[0]); return 1; }

    const std::string rec_path    = argv[1];
    const std::string cam_dir     = argv[2];
    const std::string out_path    = argv[3];

    bool bypass_tracker = false;
    bool log_tracking   = false;
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bypass-tracker") bypass_tracker = true;
        else if (arg == "--log-tracking") log_tracking = true;
        else { std::fprintf(stderr, "unknown option: %s\n", arg.c_str()); usage(argv[0]); return 1; }
    }

    try {
        // Load recording
        std::printf("[replay] loading %s\n", rec_path.c_str());
        const RecordingData rec = load_recording(rec_path);
        std::printf("[replay] %zu poses, %zu frames\n", rec.poses.size(), rec.frames.size());

        // Load camera configs (intrinsics/extrinsics for PoseEstimator)
        const CameraLoadResult cam = load_camera_configs(cam_dir);

        // Build output sink
        JsonlOutput output(out_path);

        // Build processor — pose buffer sized to hold all recorded poses
        DetectionProcessor::Config proc_cfg;
        proc_cfg.pose_cameras         = cam.pose_cameras;
        proc_cfg.bypass_tracker       = bypass_tracker;
        proc_cfg.log_tracking         = log_tracking;
        proc_cfg.log_path             = out_path + ".tracker_log.csv";
        proc_cfg.pose_buffer_capacity = rec.poses.size() > 0 ? rec.poses.size() : PoseBuffer::N;

        DetectionProcessor processor(std::move(proc_cfg), output);

        // Replay causally (5.17): before each frame, push only the poses whose time is at or
        // before that frame's capture time, so a frame never "sees" a pose that in live
        // operation arrived after it. Poses are pushed in recorded order (sorted by time).
        // Frames drive process() with their recorded capture time so empty-frame LOST/miss
        // timing matches live instead of collapsing in a tight loop.
        size_t pose_i = 0;
        for (const auto& frame : rec.frames) {
            const uint64_t frame_ns = frame.frame_capture_ns;
            while (pose_i < rec.poses.size()
                   && (frame_ns == 0 || rec.poses[pose_i].jetson_recv_ns <= frame_ns)) {
                processor.push_pose(rec.poses[pose_i]);
                ++pose_i;
            }
            processor.process(frame.dets, frame_ns);
        }
        // Flush any remaining poses (e.g. legacy recordings with no frame timestamps).
        for (; pose_i < rec.poses.size(); ++pose_i)
            processor.push_pose(rec.poses[pose_i]);

        std::printf("[replay] done → %s\n", out_path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[replay] error: %s\n", e.what());
        return 1;
    }
    return 0;
}
