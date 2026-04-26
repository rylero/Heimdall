#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "tracker/jpda.h"
#include "tracker/kalman.h"

using Catch::Matchers::WithinAbs;

static JpdaConfig default_cfg() {
    return {.gate_distance = 1.0f, .clutter_density = 1.0f, .p_detection = 0.9f};
}

TEST_CASE("empty tracks -- all detections returned as unassociated", "[jpda]") {
    std::vector<Track> tracks;
    std::vector<FieldDetection> dets = {{0, 1.f, 2.f, 0.9f}, {0, 5.f, 5.f, 0.8f}};
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.size() == 2);
    REQUIRE(unassoc[0] == 0);
    REQUIRE(unassoc[1] == 1);
}

TEST_CASE("empty detections -- all tracks get frames_missed incremented", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    jpda_update(tracks, {}, 0.02, default_cfg());
    REQUIRE(tracks[0].frames_missed == 1);
    REQUIRE(tracks[0].frames_seen   == 1);
}

TEST_CASE("nearby detection is associated -- not returned as unassociated", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    std::vector<FieldDetection> dets = {{0, 0.1f, 0.1f, 0.9f}};
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.empty());
}

TEST_CASE("far detection is unassociated -- outside gate", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    std::vector<FieldDetection> dets = {{0, 5.f, 5.f, 0.9f}};
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.size() == 1);
    REQUIRE(unassoc[0] == 0);
}

TEST_CASE("associated detection updates track state toward measurement", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    float meas_x = 0.5f, meas_y = 0.3f;
    std::vector<FieldDetection> dets = {{0, meas_x, meas_y, 0.9f}};
    float x_before = tracks[0].state[0];
    jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(std::abs(tracks[0].state[0] - meas_x) < std::abs(x_before - meas_x));
}

TEST_CASE("associated detection increments frames_seen, resets frames_missed", "[jpda]") {
    std::vector<Track> tracks = { make_track(1, 0, 0.f, 0.f, 1.f, 0.0) };
    tracks[0].frames_missed = 2;
    std::vector<FieldDetection> dets = {{0, 0.1f, 0.f, 0.9f}};
    jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(tracks[0].frames_seen   == 2);
    REQUIRE(tracks[0].frames_missed == 0);
}

TEST_CASE("two well-separated tracks -- each associated with its own detection", "[jpda]") {
    std::vector<Track> tracks = {
        make_track(1, 0,  0.f,  0.f, 1.f, 0.0),
        make_track(2, 0, 10.f, 10.f, 1.f, 0.0),
    };
    std::vector<FieldDetection> dets = {
        {0, 0.1f,  0.1f,  0.9f},
        {0, 10.1f, 10.1f, 0.9f},
    };
    auto unassoc = jpda_update(tracks, dets, 0.02, default_cfg());
    REQUIRE(unassoc.empty());
    REQUIRE(tracks[0].frames_missed == 0);
    REQUIRE(tracks[1].frames_missed == 0);
}
