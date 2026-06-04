#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "nvdsinfer_custom_impl.h"
#include <vector>
#include <cstring>

extern "C" bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const&,
    NvDsInferNetworkInfo            const&,
    NvDsInferParseDetectionParams   const&,
    std::vector<NvDsInferParseObjectInfo>&);

// Build a minimal transposed YOLO output buffer: [4+nc, num_anchors]
// One anchor at (cx, cy, w, h) with class score, rest near zero.
static std::vector<float> make_yolo_buffer(
    int num_anchors, int num_classes,
    int anchor_idx, float cx, float cy, float bw, float bh, float score)
{
    int rows = 4 + num_classes;
    std::vector<float> buf(rows * num_anchors, 0.f);
    buf[0 * num_anchors + anchor_idx] = cx;
    buf[1 * num_anchors + anchor_idx] = cy;
    buf[2 * num_anchors + anchor_idx] = bw;
    buf[3 * num_anchors + anchor_idx] = bh;
    buf[4 * num_anchors + anchor_idx] = score;
    return buf;
}

TEST_CASE("YOLO parser outputs pixel-space coords without re-scaling", "[parser]") {
    // Regression: parser previously multiplied by networkInfo.width/height,
    // double-scaling coords that YOLO11 already outputs in pixel space.
    const int   NUM_ANCHORS  = 8400;
    const int   NUM_CLASSES  = 1;
    const float NET_W = 640.f, NET_H = 640.f;

    // Anchor in the middle of the image, box ~quarter the image size
    const float CX = 320.f, CY = 240.f, BW = 160.f, BH = 120.f, SCORE = 0.8f;
    auto buf = make_yolo_buffer(NUM_ANCHORS, NUM_CLASSES, 0, CX, CY, BW, BH, SCORE);

    NvDsInferLayerInfo layer{};
    std::memcpy(layer.layerName, "output0", sizeof("output0"));
    layer.inferDims.numDims    = 2;
    layer.inferDims.d[0]       = 4 + NUM_CLASSES;
    layer.inferDims.d[1]       = NUM_ANCHORS;
    layer.buffer               = buf.data();

    NvDsInferNetworkInfo net{};
    net.width  = static_cast<unsigned int>(NET_W);
    net.height = static_cast<unsigned int>(NET_H);

    NvDsInferParseDetectionParams params{};
    params.perClassThreshold = {0.5f};
    params.numClassesConfigured = 1;

    std::vector<NvDsInferParseObjectInfo> objects;
    bool ok = NvDsInferParseCustomYolo({layer}, net, params, objects);

    REQUIRE(ok);
    REQUIRE(objects.size() == 1);

    const auto& o = objects[0];
    // Coordinates must be in pixel space (0..NET_W/H), not re-scaled
    CHECK_THAT(o.left,   Catch::Matchers::WithinAbs(CX - BW / 2.f, 0.01f));
    CHECK_THAT(o.top,    Catch::Matchers::WithinAbs(CY - BH / 2.f, 0.01f));
    CHECK_THAT(o.width,  Catch::Matchers::WithinAbs(BW,             0.01f));
    CHECK_THAT(o.height, Catch::Matchers::WithinAbs(BH,             0.01f));
    CHECK(o.left   < NET_W);
    CHECK(o.top    < NET_H);
    CHECK(o.width  < NET_W);
    CHECK(o.height < NET_H);
    CHECK_THAT(o.detectionConfidence, Catch::Matchers::WithinAbs(SCORE, 0.001f));
    CHECK(o.classId == 0);
}

TEST_CASE("YOLO parser skips anchors below threshold", "[parser]") {
    const int NUM_ANCHORS = 8400, NUM_CLASSES = 1;
    auto buf = make_yolo_buffer(NUM_ANCHORS, NUM_CLASSES, 42, 320.f, 240.f, 100.f, 100.f, 0.3f);

    NvDsInferLayerInfo layer{};
    std::memcpy(layer.layerName, "output0", sizeof("output0"));
    layer.inferDims.d[0] = 5; layer.inferDims.d[1] = NUM_ANCHORS;
    layer.buffer = buf.data();

    NvDsInferNetworkInfo net{}; net.width = 640; net.height = 640;
    NvDsInferParseDetectionParams params{}; params.perClassThreshold = {0.5f};

    std::vector<NvDsInferParseObjectInfo> objects;
    NvDsInferParseCustomYolo({layer}, net, params, objects);
    REQUIRE(objects.empty());
}

TEST_CASE("YOLO parser returns false when output0 layer missing", "[parser]") {
    NvDsInferLayerInfo layer{};
    std::memcpy(layer.layerName, "wrong_name", sizeof("wrong_name"));
    layer.inferDims.d[0] = 5; layer.inferDims.d[1] = 100;
    std::vector<float> buf(500, 0.f);
    layer.buffer = buf.data();

    NvDsInferNetworkInfo net{}; net.width = 640; net.height = 640;
    NvDsInferParseDetectionParams params{}; params.perClassThreshold = {0.5f};

    std::vector<NvDsInferParseObjectInfo> objects;
    bool ok = NvDsInferParseCustomYolo({layer}, net, params, objects);
    REQUIRE_FALSE(ok);
}
