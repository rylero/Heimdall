#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tracker/kalman.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("predict with dt=0 does not change position", "[kalman]") {
    Track t = make_track(1, 0, 3.f, 4.f, 1.f, 0.0);
    t.state[2] = 1.f; t.state[3] = -1.f;
    kalman_predict(t, 0.0);
    REQUIRE_THAT(t.state[0], WithinAbs(3.f, 1e-5f));
    REQUIRE_THAT(t.state[1], WithinAbs(4.f, 1e-5f));
}

TEST_CASE("predict moves position by velocity * dt", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0);
    t.state[2] = 2.f;   // vx = 2 m/s
    t.state[3] = -1.f;  // vy = -1 m/s
    kalman_predict(t, 0.5);
    REQUIRE_THAT(t.state[0], WithinAbs(1.f,  1e-4f));
    REQUIRE_THAT(t.state[1], WithinAbs(-0.5f, 1e-4f));
}

TEST_CASE("predict increases position variance", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0);
    float p00_before = t.cov[0];
    kalman_predict(t, 0.02);
    REQUIRE(t.cov[0] > p00_before);
}

TEST_CASE("update with weight=0 is a no-op", "[kalman]") {
    Track t = make_track(1, 0, 2.f, 3.f, 1.f, 0.0);
    auto state_before = t.state;
    auto cov_before   = t.cov;
    kalman_update_combined(t, 100.f, 100.f, 0.f);
    REQUIRE(t.state == state_before);
    REQUIRE(t.cov   == cov_before);
}

TEST_CASE("update reduces position variance", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0);
    float p00_before = t.cov[0];
    kalman_update_combined(t, 1.f, 1.f, 1.f);
    REQUIRE(t.cov[0] < p00_before);
}

TEST_CASE("update pulls state toward measurement", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0);
    // Large P (high uncertainty) so Kalman gain K ≈ 1 and state moves to measurement.
    // P[0,0] = 100 >> R = 0.005 => K ≈ 100/100.005 ≈ 0.99995
    t.cov = {100.f, 0.f, 0.f,  0.f,
             0.f,  100.f, 0.f, 0.f,
             0.f,  0.f,  100.f, 0.f,
             0.f,  0.f,  0.f,  100.f};
    // Innovation = meas - pred = (5,3) - (0,0)
    kalman_update_combined(t, 5.f, 3.f, 1.f);
    REQUIRE_THAT(t.state[0], WithinAbs(5.f, 0.1f));
    REQUIRE_THAT(t.state[1], WithinAbs(3.f, 0.1f));
}

TEST_CASE("make_track initializes at given position with zero velocity", "[kalman]") {
    Track t = make_track(42, 3, 1.5f, -2.f, 0.8f, 1.23);
    REQUIRE(t.id          == 42);
    REQUIRE(t.class_id    == 3);
    REQUIRE_THAT(t.state[0], WithinAbs(1.5f, 1e-6f));
    REQUIRE_THAT(t.state[1], WithinAbs(-2.f, 1e-6f));
    REQUIRE_THAT(t.state[2], WithinAbs(0.f,  1e-6f));
    REQUIRE_THAT(t.state[3], WithinAbs(0.f,  1e-6f));
    REQUIRE(t.status        == TrackStatus::TENTATIVE);
    REQUIRE(t.frames_seen   == 1);
    REQUIRE(t.frames_missed == 0);
}
