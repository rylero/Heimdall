#include <catch2/catch_test_macros.hpp>
#include "pipeline/camera_source.h"

TEST_CASE("USB source contains v4l2src and device", "[camera]") {
    CameraConfig cfg{0, CameraType::USB, "/dev/video0"};
    auto s = build_source_description(cfg);
    REQUIRE(s.find("v4l2src") != std::string::npos);
    REQUIRE(s.find("device=/dev/video0") != std::string::npos);
}

TEST_CASE("USB source contains resolution", "[camera]") {
    CameraConfig cfg{0, CameraType::USB, "/dev/video0", 640, 480};
    auto s = build_source_description(cfg);
    REQUIRE(s.find("width=640") != std::string::npos);
    REQUIRE(s.find("height=480") != std::string::npos);
}

TEST_CASE("CSI source contains nvarguscamerasrc and sensor-id", "[camera]") {
    CameraConfig cfg{1, CameraType::CSI, "0"};
    auto s = build_source_description(cfg);
    REQUIRE(s.find("nvarguscamerasrc") != std::string::npos);
    REQUIRE(s.find("sensor-id=0") != std::string::npos);
}

TEST_CASE("Unknown type throws invalid_argument", "[camera]") {
    CameraConfig cfg{0, static_cast<CameraType>(99), "/dev/video0"};
    REQUIRE_THROWS_AS(build_source_description(cfg), std::invalid_argument);
}
