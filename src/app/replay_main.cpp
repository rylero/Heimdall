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
        "  --bypass-tracker        pass raw detections through without JPDAF\n"
        "  --log-tracking          write tracker_log.csv alongside output\n"
        "  Tracker params (override ObjectTrackerConfig defaults; enables tuning sweeps):\n"
        "  --confirm-frames  N     frames before a new track is CONFIRMED\n"
        "  --loss-frames     N     frames without a match before a track is LOST\n"
        "  --gate-distance   F     Mahalanobis/gate distance (m)\n"
        "  --clutter-density F     expected false-positive density per m^2\n"
        "  --p-detection     F     probability a real object is detected each frame\n"
        "  --meas-noise-r    F     measurement noise variance (m^2)\n"
        "  --process-noise-q F     process noise intensity\n"
        "  --pos-cov-floor   F     minimum position covariance diagonal\n"
        "  --filter-model    M     cp | cv | ca\n",
        argv0);
}

int main(int argc, char** argv) {
    if (argc < 4) { usage(argv[0]); return 1; }

    const std::string rec_path    = argv[1];
    const std::string cam_dir     = argv[2];
    const std::string out_path    = argv[3];

    bool bypass_tracker = false;
    bool log_tracking   = false;
    ObjectTracker::Config tracker;  // start from defaults, override via flags

    // Helper: read the value that follows a --flag (returns "" and flags error if missing).
    auto next_val = [&](int& i) -> std::string {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); return ""; }
        return argv[++i];
    };
    bool arg_error = false;
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if      (arg == "--bypass-tracker")  bypass_tracker = true;
        else if (arg == "--log-tracking")    log_tracking   = true;
        else if (arg == "--confirm-frames")  tracker.confirmation_frames = std::atoi(next_val(i).c_str());
        else if (arg == "--loss-frames")     tracker.loss_frames         = std::atoi(next_val(i).c_str());
        else if (arg == "--gate-distance")   tracker.gate_distance       = std::atof(next_val(i).c_str());
        else if (arg == "--clutter-density") tracker.clutter_density     = std::atof(next_val(i).c_str());
        else if (arg == "--p-detection")     tracker.p_detection         = std::atof(next_val(i).c_str());
        else if (arg == "--meas-noise-r")    tracker.meas_noise_r        = std::atof(next_val(i).c_str());
        else if (arg == "--process-noise-q") tracker.process_noise_q     = std::atof(next_val(i).c_str());
        else if (arg == "--pos-cov-floor")   tracker.pos_cov_floor       = std::atof(next_val(i).c_str());
        else if (arg == "--filter-model") {
            const std::string m = next_val(i);
            if      (m == "cp") tracker.filter_model = FilterModel::CONSTANT_POSITION;
            else if (m == "cv") tracker.filter_model = FilterModel::CONSTANT_VELOCITY;
            else if (m == "ca") tracker.filter_model = FilterModel::CONSTANT_ACCELERATION;
            else { std::fprintf(stderr, "unknown filter-model: %s\n", m.c_str()); arg_error = true; }
        }
        else { std::fprintf(stderr, "unknown option: %s\n", arg.c_str()); arg_error = true; }
    }
    if (arg_error) { usage(argv[0]); return 1; }

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
        proc_cfg.tracker              = tracker;
        proc_cfg.bypass_tracker       = bypass_tracker;
        proc_cfg.log_tracking         = log_tracking;
        proc_cfg.log_path             = out_path + ".tracker_log.csv";
        proc_cfg.pose_buffer_capacity = rec.poses.size() > 0 ? rec.poses.size() : PoseBuffer::N;

        std::printf("[replay] tracker: confirm=%d loss=%d gate=%.2f clutter=%.2f "
                    "p_det=%.2f R=%.3f Q=%.2f floor=%.2f model=%d\n",
                    tracker.confirmation_frames, tracker.loss_frames, tracker.gate_distance,
                    tracker.clutter_density, tracker.p_detection, tracker.meas_noise_r,
                    tracker.process_noise_q, tracker.pos_cov_floor,
                    static_cast<int>(tracker.filter_model));

        DetectionProcessor processor(std::move(proc_cfg), output);

        // Preload all poses before processing frames (no eviction risk)
        for (const auto& p : rec.poses)
            processor.push_pose(p);

        // Replay frames in recorded order
        for (const auto& frame : rec.frames)
            processor.process(frame.dets);

        std::printf("[replay] done → %s\n", out_path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[replay] error: %s\n", e.what());
        return 1;
    }
    return 0;
}
