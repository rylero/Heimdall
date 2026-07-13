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

TEST_CASE("USB source hw_decode=true uses nvv4l2decoder", "[camera]") {
    CameraConfig cfg{.id=0, .type=CameraType::USB, .device="/dev/video0", .width=640, .height=480, .fps=60, .hw_decode=true};
    auto s = build_source_description(cfg);
    REQUIRE(s.find("nvv4l2decoder") != std::string::npos);
    REQUIRE(s.find("mjpeg=1") != std::string::npos);
    REQUIRE(s.find("jpegdec") == std::string::npos);
    REQUIRE(s.find("nvvidconv") != std::string::npos);
}

TEST_CASE("USB source hw_decode=false uses CPU jpegdec", "[camera]") {
    CameraConfig cfg{.id=1, .type=CameraType::USB, .device="/dev/video2", .width=640, .height=480, .fps=30, .hw_decode=false};
    auto s = build_source_description(cfg);
    REQUIRE(s.find("jpegdec") != std::string::npos);
    REQUIRE(s.find("nvv4l2decoder") == std::string::npos);
    REQUIRE(s.find("nvvidconv") != std::string::npos);
}

TEST_CASE("CSI source contains nvarguscamerasrc and sensor-id", "[camera]") {
    CameraConfig cfg{1, CameraType::CSI, "0"};
    auto s = build_source_description(cfg);
    REQUIRE(s.find("nvarguscamerasrc") != std::string::npos);
    REQUIRE(s.find("sensor-id=0") != std::string::npos);
}

TEST_CASE("TEST source uses videotestsrc with nvvidconv for NVMM output", "[camera]") {
    CameraConfig cfg{2, CameraType::TEST, "", 640, 480, 100};
    auto s = build_source_description(cfg);
    REQUIRE(s.find("videotestsrc") != std::string::npos);
    REQUIRE(s.find("is-live=true") != std::string::npos);
    REQUIRE(s.find("nvvidconv") != std::string::npos);
    REQUIRE(s.find("width=640") != std::string::npos);
    REQUIRE(s.find("height=480") != std::string::npos);
}

TEST_CASE("Mirror camera has mirror_of set to source id", "[camera]") {
    CameraConfig real{.id=0, .type=CameraType::USB, .device="/dev/video0", .width=640, .height=480, .fps=100};
    CameraConfig mirror{.id=2, .type=CameraType::USB, .device="", .width=640, .height=480, .fps=100, .mirror_of=0};
    REQUIRE(real.mirror_of == -1);
    REQUIRE(mirror.mirror_of == 0);
}

TEST_CASE("rotation maps to the correct nvvidconv flip-method", "[camera][rotation]") {
    auto method = [](int rot) {
        CameraConfig cfg{.id=0, .type=CameraType::USB, .device="/dev/video0",
                         .width=640, .height=480, .fps=60, .rotation=rot};
        return build_source_description(cfg);
    };
    REQUIRE(method(0)  .find("flip-method=0") != std::string::npos);
    REQUIRE(method(90) .find("flip-method=3") != std::string::npos);  // clockwise 90
    REQUIRE(method(180).find("flip-method=2") != std::string::npos);
    REQUIRE(method(270).find("flip-method=1") != std::string::npos);  // == ccw 90
}

TEST_CASE("Unknown type throws invalid_argument", "[camera]") {
    CameraConfig cfg{0, static_cast<CameraType>(99), "/dev/video0"};
    REQUIRE_THROWS_AS(build_source_description(cfg), std::invalid_argument);
}
