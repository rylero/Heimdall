// Custom DeepStream bbox parser for YOLOv8/v11 TensorRT output.
//
// Expected output layer (verify with trtexec --listLayers or netron):
//   output0  [1, 4+nc, num_anchors]  — transposed layout
//     rows 0-3:     cx, cy, w, h  (normalized 0..1)
//     rows 4..4+nc: class scores  (sigmoid already applied by exporter)
//
// If your export uses a different layer name update OUTPUT_LAYER below.

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

static constexpr const char* OUTPUT_LAYER = "output0";

extern "C" bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo            const& networkInfo,
    NvDsInferParseDetectionParams   const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList
) {
    const float* out         = nullptr;
    int          num_anchors = 0;
    int          row_stride  = 0; // = 4 + num_classes

    for (const auto& layer : outputLayersInfo) {
        if (std::string(layer.layerName) == OUTPUT_LAYER) {
            // dims: [4+nc, num_anchors]  (batch dim stripped by nvinfer)
            row_stride  = layer.inferDims.d[0];
            num_anchors = layer.inferDims.d[1];
            out         = static_cast<const float*>(layer.buffer);
            break;
        }
    }

    if (!out || num_anchors <= 0 || row_stride < 5) {
        std::fprintf(stderr, "[yolo_parser] layer '%s' not found or bad dims (stride=%d anchors=%d)\n",
                     OUTPUT_LAYER, row_stride, num_anchors);
        return false;
    }
    if (detectionParams.perClassThreshold.empty())  return false;

    const int num_classes = row_stride - 4;

    // Print max score seen this call to diagnose threshold issues (every 100 calls)
    static std::atomic<int> s_call{0};
    bool do_log = (s_call.fetch_add(1) % 100 == 0);
    float max_score_seen = 0.f;

    for (int a = 0; a < num_anchors; ++a) {
        // Transposed: value at row r, anchor a → out[r * num_anchors + a]
        float best_score = -1.f;
        int   best_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = out[(4 + c) * num_anchors + a];
            if (score > best_score) { best_score = score; best_class = c; }
        }
        if (do_log && best_score > max_score_seen) max_score_seen = best_score;

        float threshold = (best_class < static_cast<int>(detectionParams.perClassThreshold.size()))
            ? detectionParams.perClassThreshold[best_class]
            : detectionParams.perClassThreshold[0];

        if (best_score < threshold) continue;

        float cx = out[0 * num_anchors + a];
        float cy = out[1 * num_anchors + a];
        float bw = out[2 * num_anchors + a];
        float bh = out[3 * num_anchors + a];

        NvDsInferParseObjectInfo obj{};
        obj.classId             = static_cast<unsigned int>(best_class);
        obj.detectionConfidence = best_score;
        obj.left   = (cx - bw / 2.f) * networkInfo.width;
        obj.top    = (cy - bh / 2.f) * networkInfo.height;
        obj.width  = bw * networkInfo.width;
        obj.height = bh * networkInfo.height;
        objectList.push_back(obj);
    }
    if (do_log) {
        std::fprintf(stderr, "[yolo_parser] detections=%d  max_score=%.4f  threshold=%.2f  net=%dx%d\n",
                     static_cast<int>(objectList.size()), max_score_seen,
                     detectionParams.perClassThreshold.empty() ? 0.f : detectionParams.perClassThreshold[0],
                     networkInfo.width, networkInfo.height);
        for (int i = 0; i < std::min((int)objectList.size(), 3); ++i) {
            const auto& o = objectList[i];
            std::fprintf(stderr, "  [%d] conf=%.3f cls=%u l=%.1f t=%.1f w=%.1f h=%.1f\n",
                         i, o.detectionConfidence, o.classId, o.left, o.top, o.width, o.height);
        }
    }
    return true;
}
