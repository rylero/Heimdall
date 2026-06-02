#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tracker/kalman.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("predict with dt=0 does not change position", "[kalman]") {
    Track t = make_track(1, 0, 3.f, 4.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    t.state[2] = 1.f; t.state[3] = -1.f;
    kalman_predict(t, 0.0);
    REQUIRE_THAT(t.state[0], WithinAbs(3.f, 1e-5f));
    REQUIRE_THAT(t.state[1], WithinAbs(4.f, 1e-5f));
}

TEST_CASE("predict moves position by velocity * dt", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    t.state[2] = 2.f;
    t.state[3] = -1.f;
    kalman_predict(t, 0.5);
    REQUIRE_THAT(t.state[0], WithinAbs(1.f,  1e-4f));
    REQUIRE_THAT(t.state[1], WithinAbs(-0.5f, 1e-4f));
}

TEST_CASE("predict increases position variance", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    float p00_before = t.cov[0];
    kalman_predict(t, 0.02);
    REQUIRE(t.cov[0] > p00_before);
}

TEST_CASE("update with weight=0 is a no-op", "[kalman]") {
    Track t = make_track(1, 0, 2.f, 3.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    auto state_before = t.state;
    auto cov_before   = t.cov;
    kalman_update_combined(t, 100.f, 100.f, 0.f);
    REQUIRE(t.state == state_before);
    REQUIRE(t.cov   == cov_before);
}

TEST_CASE("update reduces position variance", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    float p00_before = t.cov[0];
    kalman_update_combined(t, 1.f, 1.f, 1.f);
    REQUIRE(t.cov[0] < p00_before);
}

TEST_CASE("update pulls state toward measurement", "[kalman]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_VELOCITY);
    t.cov = {100.f, 0.f, 0.f,  0.f,
             0.f,  100.f, 0.f, 0.f,
             0.f,  0.f,  100.f, 0.f,
             0.f,  0.f,  0.f,  100.f};
    kalman_update_combined(t, 5.f, 3.f, 1.f);
    REQUIRE_THAT(t.state[0], WithinAbs(5.f, 0.1f));
    REQUIRE_THAT(t.state[1], WithinAbs(3.f, 0.1f));
}

TEST_CASE("make_track initializes at given position with zero velocity", "[kalman]") {
    Track t = make_track(42, 3, 1.5f, -2.f, 0.8f, 1.23, FilterModel::CONSTANT_VELOCITY);
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

// ─── Constant-Position (CP) tests ─────────────────────────────────────────

TEST_CASE("CP: predict does not change position", "[kalman][cp]") {
    Track t = make_track(1, 0, 3.f, 4.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    kalman_predict(t, 0.1);
    REQUIRE_THAT(t.state[0], WithinAbs(3.f, 1e-5f));
    REQUIRE_THAT(t.state[1], WithinAbs(4.f, 1e-5f));
}

TEST_CASE("CP: predict increases position variance", "[kalman][cp]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    float p00_before = t.cov[0*2+0];
    kalman_predict(t, 0.02);
    REQUIRE(t.cov[0*2+0] > p00_before);
}

TEST_CASE("CP: update reduces position variance", "[kalman][cp]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    float p00_before = t.cov[0*2+0];
    kalman_update_combined(t, 1.f, 1.f, 1.f);
    REQUIRE(t.cov[0*2+0] < p00_before);
}

TEST_CASE("CP: update pulls state toward measurement", "[kalman][cp]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    t.cov[0*2+0] = 100.f;
    t.cov[1*2+1] = 100.f;
    kalman_update_combined(t, 5.f, 3.f, 1.f);
    REQUIRE_THAT(t.state[0], WithinAbs(5.f, 0.1f));
    REQUIRE_THAT(t.state[1], WithinAbs(3.f, 0.1f));
}

TEST_CASE("CP: update with weight=0 is a no-op", "[kalman][cp]") {
    Track t = make_track(1, 0, 2.f, 3.f, 1.f, 0.0, FilterModel::CONSTANT_POSITION);
    auto state_before = t.state;
    auto cov_before   = t.cov;
    kalman_update_combined(t, 100.f, 100.f, 0.f);
    REQUIRE(t.state == state_before);
    REQUIRE(t.cov   == cov_before);
}

TEST_CASE("CP: make_track initializes with zero velocity slots", "[kalman][cp]") {
    Track t = make_track(1, 0, 2.f, -1.f, 0.9f, 0.5, FilterModel::CONSTANT_POSITION);
    REQUIRE(t.model == FilterModel::CONSTANT_POSITION);
    REQUIRE_THAT(t.state[0], WithinAbs(2.f,  1e-6f));
    REQUIRE_THAT(t.state[1], WithinAbs(-1.f, 1e-6f));
    REQUIRE_THAT(t.state[2], WithinAbs(0.f,  1e-6f));  // unused vx
    REQUIRE_THAT(t.state[3], WithinAbs(0.f,  1e-6f));  // unused vy
    REQUIRE_THAT(t.cov[0*2+0], WithinAbs(10.f, 1e-5f));
    REQUIRE_THAT(t.cov[1*2+1], WithinAbs(10.f, 1e-5f));
    REQUIRE_THAT(t.cov[1*2+0], WithinAbs(0.f,  1e-5f));  // off-diagonal zero
}

// ─── Constant-Acceleration (CA) tests ─────────────────────────────────────

TEST_CASE("CA: predict advances position with velocity and acceleration", "[kalman][ca]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    t.state[2] = 1.f;  // vx = 1 m/s
    t.state[4] = 2.f;  // ax = 2 m/s^2
    kalman_predict(t, 1.0);
    // x = 0 + 1*1 + 0.5*2*1^2 = 2.0
    REQUIRE_THAT(t.state[0], WithinAbs(2.f,  1e-4f));
    // vx = 1 + 2*1 = 3.0
    REQUIRE_THAT(t.state[2], WithinAbs(3.f,  1e-4f));
    // ax unchanged
    REQUIRE_THAT(t.state[4], WithinAbs(2.f,  1e-4f));
}

TEST_CASE("CA: predict increases position variance", "[kalman][ca]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    float p00_before = t.cov[0*6+0];
    kalman_predict(t, 0.02);
    REQUIRE(t.cov[0*6+0] > p00_before);
}

TEST_CASE("CA: update reduces position variance", "[kalman][ca]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    float p00_before = t.cov[0*6+0];
    kalman_update_combined(t, 1.f, 1.f, 1.f);
    REQUIRE(t.cov[0*6+0] < p00_before);
}

TEST_CASE("CA: update pulls position toward measurement", "[kalman][ca]") {
    Track t = make_track(1, 0, 0.f, 0.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    t.cov[0*6+0] = 100.f;
    t.cov[1*6+1] = 100.f;
    kalman_update_combined(t, 5.f, 3.f, 1.f);
    REQUIRE_THAT(t.state[0], WithinAbs(5.f, 0.1f));
    REQUIRE_THAT(t.state[1], WithinAbs(3.f, 0.1f));
}

TEST_CASE("CA: update with weight=0 is a no-op", "[kalman][ca]") {
    Track t = make_track(1, 0, 2.f, 3.f, 1.f, 0.0, FilterModel::CONSTANT_ACCELERATION);
    auto state_before = t.state;
    auto cov_before   = t.cov;
    kalman_update_combined(t, 100.f, 100.f, 0.f);
    REQUIRE(t.state == state_before);
    REQUIRE(t.cov   == cov_before);
}

TEST_CASE("CA: make_track initializes with zero vel/accel", "[kalman][ca]") {
    Track t = make_track(1, 0, 1.f, 2.f, 0.9f, 0.5, FilterModel::CONSTANT_ACCELERATION);
    REQUIRE(t.model == FilterModel::CONSTANT_ACCELERATION);
    REQUIRE_THAT(t.state[0], WithinAbs(1.f, 1e-6f));
    REQUIRE_THAT(t.state[1], WithinAbs(2.f, 1e-6f));
    for (int i = 2; i < 6; ++i)
        REQUIRE_THAT(t.state[i], WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT(t.cov[0*6+0], WithinAbs(10.f,  1e-5f));
    REQUIRE_THAT(t.cov[1*6+1], WithinAbs(10.f,  1e-5f));
    REQUIRE_THAT(t.cov[2*6+2], WithinAbs(1.f,   1e-5f));
    REQUIRE_THAT(t.cov[3*6+3], WithinAbs(1.f,   1e-5f));
    REQUIRE_THAT(t.cov[4*6+4], WithinAbs(0.1f,  1e-5f));
    REQUIRE_THAT(t.cov[5*6+5], WithinAbs(0.1f,  1e-5f));
}
