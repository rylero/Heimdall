#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "app/pose_buffer.h"

using Catch::Matchers::WithinAbs;

// Helper: build a TimestampedPose with a known reception time and x/y values.
static CommLayer::TimestampedPose make_tp(uint64_t recv_ns, float x, float y, float heading = 0.f) {
    return CommLayer::TimestampedPose{RobotPose{x, y, heading, 0}, recv_ns};
}

// ── Empty buffer ───────────────────────────────────────────────────────────────

TEST_CASE("PoseBuffer: empty returns zero pose", "[pose_buffer]") {
    PoseBuffer buf;
    RobotPose p = buf.closest(1'000'000ULL);
    REQUIRE_THAT(p.x,       WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT(p.y,       WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT(p.heading, WithinAbs(0.f, 1e-6f));
}

// ── Single entry ──────────────────────────────────────────────────────────────

TEST_CASE("PoseBuffer: single entry always returned", "[pose_buffer]") {
    PoseBuffer buf;
    buf.push(make_tp(1'000'000ULL, 3.f, 4.f, 1.f));

    // target far before
    RobotPose p1 = buf.closest(0ULL);
    REQUIRE_THAT(p1.x, WithinAbs(3.f, 1e-5f));
    REQUIRE_THAT(p1.y, WithinAbs(4.f, 1e-5f));

    // target exact
    RobotPose p2 = buf.closest(1'000'000ULL);
    REQUIRE_THAT(p2.x, WithinAbs(3.f, 1e-5f));

    // target far after
    RobotPose p3 = buf.closest(999'999'999'999ULL);
    REQUIRE_THAT(p3.x, WithinAbs(3.f, 1e-5f));
}

// ── Two entries: picks closer ─────────────────────────────────────────────────

TEST_CASE("PoseBuffer: two entries picks nearest", "[pose_buffer]") {
    PoseBuffer buf;
    buf.push(make_tp(10'000'000ULL, 1.f, 0.f)); // t=10 ms
    buf.push(make_tp(30'000'000ULL, 2.f, 0.f)); // t=30 ms

    // target at 15 ms — closer to first (diff 5 vs 15)
    RobotPose p1 = buf.closest(15'000'000ULL);
    REQUIRE_THAT(p1.x, WithinAbs(1.f, 1e-5f));

    // target at 25 ms — closer to second (diff 5 vs 15)
    RobotPose p2 = buf.closest(25'000'000ULL);
    REQUIRE_THAT(p2.x, WithinAbs(2.f, 1e-5f));

    // exact match first
    RobotPose p3 = buf.closest(10'000'000ULL);
    REQUIRE_THAT(p3.x, WithinAbs(1.f, 1e-5f));

    // exact match second
    RobotPose p4 = buf.closest(30'000'000ULL);
    REQUIRE_THAT(p4.x, WithinAbs(2.f, 1e-5f));
}

// ── Target before all entries returns oldest ──────────────────────────────────

TEST_CASE("PoseBuffer: target before all entries returns oldest", "[pose_buffer]") {
    PoseBuffer buf;
    buf.push(make_tp(100'000'000ULL, 5.f, 0.f));
    buf.push(make_tp(200'000'000ULL, 6.f, 0.f));
    buf.push(make_tp(300'000'000ULL, 7.f, 0.f));

    RobotPose p = buf.closest(0ULL);
    REQUIRE_THAT(p.x, WithinAbs(5.f, 1e-5f));
}

// ── Target after all entries returns newest ───────────────────────────────────

TEST_CASE("PoseBuffer: target after all entries returns newest", "[pose_buffer]") {
    PoseBuffer buf;
    buf.push(make_tp(100'000'000ULL, 5.f, 0.f));
    buf.push(make_tp(200'000'000ULL, 6.f, 0.f));
    buf.push(make_tp(300'000'000ULL, 7.f, 0.f));

    RobotPose p = buf.closest(999'000'000'000ULL);
    REQUIRE_THAT(p.x, WithinAbs(7.f, 1e-5f));
}

// ── Ring wrap: N+1 pushes evicts oldest ───────────────────────────────────────

TEST_CASE("PoseBuffer: ring wrap evicts oldest entry", "[pose_buffer]") {
    PoseBuffer buf;
    constexpr size_t N = PoseBuffer::N;

    // Fill exactly N slots with recv_ns = i * 1e6, x = i
    for (size_t i = 0; i < N; ++i)
        buf.push(make_tp(static_cast<uint64_t>(i) * 1'000'000ULL,
                         static_cast<float>(i), 0.f));

    // Slot 0 (recv_ns=0, x=0) is the oldest. Push one more to evict it.
    buf.push(make_tp(static_cast<uint64_t>(N) * 1'000'000ULL,
                     static_cast<float>(N), 0.f));

    // Target at recv_ns=0 — slot 0 is gone; nearest surviving is slot 1 (recv_ns=1 ms)
    RobotPose p = buf.closest(0ULL);
    REQUIRE_THAT(p.x, WithinAbs(1.f, 1e-5f));

    // Newest slot should be accessible
    RobotPose newest = buf.closest(static_cast<uint64_t>(N) * 1'000'000ULL);
    REQUIRE_THAT(newest.x, WithinAbs(static_cast<float>(N), 1e-5f));
}

// ── Many wraps: no index corruption ───────────────────────────────────────────

TEST_CASE("PoseBuffer: survives 10x ring wrap without corruption", "[pose_buffer]") {
    PoseBuffer buf;
    constexpr size_t N     = PoseBuffer::N;
    constexpr size_t total = N * 10 + 7;

    for (size_t i = 0; i < total; ++i)
        buf.push(make_tp(static_cast<uint64_t>(i) * 1'000'000ULL,
                         static_cast<float>(i), 0.f));

    // After 10 full wraps the buffer holds entries [total-N .. total-1].
    // Newest = total-1
    const float expected_newest = static_cast<float>(total - 1);
    RobotPose newest = buf.closest(static_cast<uint64_t>(total - 1) * 1'000'000ULL);
    REQUIRE_THAT(newest.x, WithinAbs(expected_newest, 1e-2f));

    // Oldest = total-N
    const float expected_oldest = static_cast<float>(total - N);
    RobotPose oldest = buf.closest(static_cast<uint64_t>(total - N) * 1'000'000ULL);
    REQUIRE_THAT(oldest.x, WithinAbs(expected_oldest, 1e-2f));
}

// ── Realistic timing: 50 Hz poses, 35 ms pipeline latency ────────────────────

TEST_CASE("PoseBuffer: selects pose matching camera capture time at 50Hz", "[pose_buffer]") {
    PoseBuffer buf;
    // Simulate 200 ms of 50 Hz poses (10 poses, 20 ms apart)
    // x encodes the pose index for easy identification
    for (int i = 0; i < 10; ++i)
        buf.push(make_tp(static_cast<uint64_t>(i) * 20'000'000ULL, static_cast<float>(i), 0.f));

    // Camera captured a frame at t=100 ms (pose index 5).
    // Pipeline latency is 35 ms, so on_detections fires at ~t=135 ms.
    // Without latency compensation, we'd grab pose at t=135 ms → index ~6.
    // With latency compensation, we look up t=100 ms → index 5.
    const uint64_t capture_ns = 100'000'000ULL;
    RobotPose p = buf.closest(capture_ns);
    REQUIRE_THAT(p.x, WithinAbs(5.f, 1e-5f));
}
