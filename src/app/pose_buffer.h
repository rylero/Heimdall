#pragma once
#include "comm/comm_layer.h"
#include <cstdint>
#include <mutex>
#include <vector>

// Ring buffer of recently received robot poses, indexed by Jetson CLOCK_MONOTONIC reception time.
// Allows on_detections() to select the pose that matches the camera frame's capture time
// rather than always using the latest (which is 20–60 ms newer than the captured image).
class PoseBuffer {
public:
    static constexpr size_t N = 64; // default capacity (~1.3 s of history at 50 Hz)

    explicit PoseBuffer(size_t capacity = N) : slots_(capacity), cap_(capacity) {}

    void push(const CommLayer::TimestampedPose& p) {
        std::lock_guard<std::mutex> lock(mu_);
        slots_[head_ % cap_] = p;
        ++head_;
        if (count_ < cap_) ++count_;
    }

    // Returns the pose whose jetson_recv_ns is closest to target_ns.
    // Returns a zero-initialized RobotPose if the buffer is empty (before first pose arrives).
    RobotPose closest(uint64_t target_ns) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (count_ == 0) return {};
        size_t   best_idx  = (head_ - count_) % cap_;
        uint64_t best_diff = UINT64_MAX;
        for (size_t i = 0; i < count_; ++i) {
            const size_t   idx  = (head_ - count_ + i) % cap_;
            const uint64_t t    = slots_[idx].jetson_recv_ns;
            const uint64_t diff = t >= target_ns ? t - target_ns : target_ns - t;
            if (diff < best_diff) { best_diff = diff; best_idx = idx; }
        }
        return slots_[best_idx].pose;
    }

    size_t capacity() const { return cap_; }

private:
    mutable std::mutex                      mu_;
    std::vector<CommLayer::TimestampedPose> slots_;
    size_t                                  cap_;
    size_t                                  head_  = 0;
    size_t                                  count_ = 0;
};
