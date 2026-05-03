// Custom DeepStream bbox parser for RFDetr DETR-style TensorRT output.
//
// Expected output layers (verify against your engine with trtexec --listLayers):
//   pred_logits  [num_queries, num_classes]  — raw logits (softmax applied here)
//   pred_boxes   [num_queries, 4]            — (cx, cy, w, h) normalized 0..1
//
// Update LOGITS_LAYER / BOXES_LAYER if your export uses different names.

#include "nvdsinfer_custom_impl.h"
#include <cmath>
#include <string>
#include <vector>

static constexpr const char* LOGITS_LAYER = "labels";
static constexpr const char* BOXES_LAYER  = "dets";

extern "C" bool NvDsInferParseCustomRFDetr(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo            const& networkInfo,
    NvDsInferParseDetectionParams   const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList
) {
    const float* logits      = nullptr;
    const float* boxes       = nullptr;
    int          num_queries = 0;
    int          num_classes = 0;

    for (const auto& layer : outputLayersInfo) {
        if (std::string(layer.layerName) == LOGITS_LAYER) {
            logits      = static_cast<const float*>(layer.buffer);
            num_queries = layer.inferDims.d[0];
            num_classes = layer.inferDims.d[1];
        } else if (std::string(layer.layerName) == BOXES_LAYER) {
            boxes = static_cast<const float*>(layer.buffer);
        }
    }

    // Return false so nvinfer logs a warning — mismatched layer names are a
    // common misconfiguration and should surface immediately.
    if (!logits || !boxes) return false;
    if (num_queries <= 0 || num_classes <= 0) return false;
    if (detectionParams.perClassThreshold.empty()) return false;

    for (int q = 0; q < num_queries; ++q) {
        const float* ql = logits + q * num_classes;

        // Numerically stable softmax over class logits
        float max_logit = ql[0];
        for (int c = 1; c < num_classes; ++c)
            if (ql[c] > max_logit) max_logit = ql[c];

        float sum_exp = 0.f;
        for (int c = 0; c < num_classes; ++c)
            sum_exp += std::exp(ql[c] - max_logit);

        float best_score = -1.f;
        int   best_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = std::exp(ql[c] - max_logit) / sum_exp;
            if (score > best_score) { best_score = score; best_class = c; }
        }

        float threshold = (best_class < static_cast<int>(detectionParams.perClassThreshold.size()))
            ? detectionParams.perClassThreshold[best_class]
            : detectionParams.perClassThreshold[0];

        if (best_score < threshold) continue;

        float cx = boxes[q * 4 + 0];
        float cy = boxes[q * 4 + 1];
        float bw = boxes[q * 4 + 2];
        float bh = boxes[q * 4 + 3];

        NvDsInferParseObjectInfo obj{};
        obj.classId             = static_cast<unsigned int>(best_class);
        obj.detectionConfidence = best_score;
        obj.left   = (cx - bw / 2.f) * networkInfo.width;
        obj.top    = (cy - bh / 2.f) * networkInfo.height;
        obj.width  = bw * networkInfo.width;
        obj.height = bh * networkInfo.height;
        objectList.push_back(obj);
    }
    return true;
}
