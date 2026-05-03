# Heimdall Milestone 1: DeepStream Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multi-camera DeepStream pipeline in C++ that loads an RFDetr INT8 TensorRT `.engine` and outputs raw `Detection` structs at ≥50 FPS per camera.

**Architecture:** C++20 GStreamer pipeline (camera sources → nvstreammux → nvinfer) with a pad probe extracting `NvDsObjectMeta` into `Detection` structs. A custom C++ shared library parses RFDetr's DETR-style output (pred_logits + pred_boxes) for nvinfer. All built with CMake + FetchContent. Containerized on `nvcr.io/nvidia/deepstream-l4t:7.0-triton`.

**Tech Stack:** C++20, CMake 3.22+ with FetchContent, NVIDIA DeepStream 7.x, GStreamer 1.x, TensorRT, Catch2 v3 (via FetchContent), Docker

> **Web UI** (config, model upload, live preview) is a separate Python FastAPI container — Milestone 5. Not in scope here.

---

## Milestone Roadmap (future plans)

| Milestone | Description |
|-----------|-------------|
| **1 (this plan)** | DeepStream multi-camera detection spike — C++ |
| 2 | Ground ray pose estimation (field-relative coordinates) |
| 3 | JPDA object tracker (global map, 80–100 objects) |
| 4 | ZeroMQ push-pull + Protobuf communication |
| 5 | Web interface — Python FastAPI container |
| 6 | Integration, tuning, Docker compose wiring |

---

## File Structure

```
CMakeLists.txt                        # Root CMake: pipeline lib + tests + subdirs
src/
  main.cpp                            # Validation entry point (FPS counter)
  pipeline/
    detection.h                       # Detection POD struct
    camera_source.h / camera_source.cpp   # CameraConfig + GStreamer source string
    probe.h / probe.cpp               # GStreamer pad probe → Detection vector
    pipeline.h / pipeline.cpp         # DeepStreamPipeline class
  models/
    bbox_parser/
      CMakeLists.txt                  # Builds librfdetr_parser.so
      rfdetr_parser.cpp               # Custom NvDsInferParseCustom for DETR output
tests/
  CMakeLists.txt
  test_camera_source.cpp              # Pure C++ — no hardware needed
config/
  infer_rfdetr.txt                    # nvinfer config template
docker/
  Dockerfile                          # DeepStream L4T container
  docker-compose.yml
models/
  .gitkeep                            # Drop .engine + labels.txt here
```

> **Testing boundary:** `test_camera_source` builds and runs on any machine with CMake + a C++20 compiler. All other targets (probe, pipeline, main) require DeepStream headers and run inside Docker on Jetson.

---

## Task 1: CMake Scaffold + Docker

**Files:**
- Create: `CMakeLists.txt`
- Create: `tests/CMakeLists.txt`
- Create: `src/models/bbox_parser/CMakeLists.txt` (stub — rfdetr_parser.cpp added in Task 4)
- Create: `models/.gitkeep`
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`

- [ ] **Step 1: Create root CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.22)
project(heimdall VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# ---------------------------------------------------------------------------
# Catch2 — unit testing
# ---------------------------------------------------------------------------
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# ---------------------------------------------------------------------------
# GStreamer (available inside DeepStream Docker image)
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)

# ---------------------------------------------------------------------------
# DeepStream paths
# ---------------------------------------------------------------------------
set(DS_ROOT "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream SDK root")
set(DS_INCLUDES "${DS_ROOT}/sources/includes")
set(DS_LIB_DIR  "${DS_ROOT}/lib")

find_library(NVDS_META_LIB     nvds_meta     PATHS ${DS_LIB_DIR} REQUIRED)
find_library(NVDSGST_META_LIB  nvdsgst_meta  PATHS ${DS_LIB_DIR} REQUIRED)

# ---------------------------------------------------------------------------
# Pipeline library
# ---------------------------------------------------------------------------
add_library(heimdall_pipeline
    src/pipeline/camera_source.cpp
    src/pipeline/probe.cpp
    src/pipeline/pipeline.cpp
)
target_include_directories(heimdall_pipeline PUBLIC
    src/
    ${DS_INCLUDES}
    ${GST_INCLUDE_DIRS}
)
target_link_libraries(heimdall_pipeline
    ${GST_LIBRARIES}
    ${NVDS_META_LIB}
    ${NVDSGST_META_LIB}
)

# ---------------------------------------------------------------------------
# Main executable (on-Jetson validation)
# ---------------------------------------------------------------------------
add_executable(heimdall src/main.cpp)
target_link_libraries(heimdall heimdall_pipeline)

# ---------------------------------------------------------------------------
# RFDetr bbox parser shared library
# ---------------------------------------------------------------------------
add_subdirectory(src/models/bbox_parser)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
```

- [ ] **Step 2: Create tests/CMakeLists.txt**

```cmake
add_executable(heimdall_tests
    test_camera_source.cpp
)
target_link_libraries(heimdall_tests
    heimdall_pipeline
    Catch2::Catch2WithMain
)
include(Catch)
catch_discover_tests(heimdall_tests)
```

- [ ] **Step 3: Create src/models/bbox_parser/CMakeLists.txt (stub)**

```cmake
# rfdetr_parser.cpp is added in Task 4.
# Stub keeps CMake happy during scaffold stage.
find_package(CUDA REQUIRED)

set(DS_INCLUDES "${DS_ROOT}/sources/includes")

add_library(rfdetr_parser SHARED)
target_include_directories(rfdetr_parser PRIVATE
    ${DS_INCLUDES}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(rfdetr_parser ${CUDA_LIBRARIES})
```

- [ ] **Step 4: Create models/.gitkeep**

Create empty file at `models/.gitkeep`.

- [ ] **Step 5: Create docker/Dockerfile**

```dockerfile
FROM nvcr.io/nvidia/deepstream-l4t:7.0-triton

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel $(nproc)

CMD ["/app/build/heimdall"]
```

- [ ] **Step 6: Create docker/docker-compose.yml**

```yaml
services:
  heimdall:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ../models:/app/models
      - ../config:/app/config
    devices:
      - /dev/video0:/dev/video0
      - /dev/video1:/dev/video1
    network_mode: host
```

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt tests/CMakeLists.txt src/models/bbox_parser/CMakeLists.txt \
        models/.gitkeep docker/
git commit -m "feat: CMake scaffold and Docker setup"
```

---

## Task 2: Camera Source Factory

**Files:**
- Create: `src/pipeline/detection.h`
- Create: `src/pipeline/camera_source.h`
- Create: `src/pipeline/camera_source.cpp`
- Create: `tests/test_camera_source.cpp`

> Pure C++ string building. No GStreamer or DeepStream headers required. Tests run on any machine.

- [ ] **Step 1: Write failing tests**

```cpp
// tests/test_camera_source.cpp
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
```

- [ ] **Step 2: Build to confirm failure (inside Docker)**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target heimdall_tests 2>&1 | head -20
```

Expected: compile error — `camera_source.h` not found.

- [ ] **Step 3: Create src/pipeline/detection.h**

```cpp
#pragma once
#include <cstdint>

struct Detection {
    int      camera_id;
    int      class_id;
    float    confidence;
    float    left;
    float    top;
    float    width;
    float    height;
    uint64_t timestamp_ns;
};
```

- [ ] **Step 4: Create src/pipeline/camera_source.h**

```cpp
#pragma once
#include <string>

enum class CameraType { USB, CSI };

struct CameraConfig {
    int         id;
    CameraType  type;
    std::string device;     // "/dev/video0" for USB, "0" for CSI sensor index
    int         width  = 640;
    int         height = 480;
    int         fps    = 60;
};

std::string build_source_description(const CameraConfig& config);
```

- [ ] **Step 5: Create src/pipeline/camera_source.cpp**

```cpp
#include "camera_source.h"
#include <sstream>
#include <stdexcept>

std::string build_source_description(const CameraConfig& cfg) {
    std::ostringstream ss;
    switch (cfg.type) {
        case CameraType::USB:
            ss << "v4l2src device=" << cfg.device
               << " ! image/jpeg"
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",framerate=" << cfg.fps << "/1"
               << " ! jpegdec ! nvvidconv"
               << " ! video/x-raw(memory:NVMM),format=NV12";
            break;
        case CameraType::CSI:
            ss << "nvarguscamerasrc sensor-id=" << cfg.device
               << " ! video/x-raw(memory:NVMM)"
               << ",width="  << cfg.width
               << ",height=" << cfg.height
               << ",format=NV12"
               << ",framerate=" << cfg.fps << "/1";
            break;
        default:
            throw std::invalid_argument("Unknown CameraType");
    }
    return ss.str();
}
```

- [ ] **Step 6: Build and run tests (inside Docker)**

```bash
cmake --build build --target heimdall_tests
cd build && ctest --output-on-failure
```

Expected:
```
Test #1: USB source contains v4l2src ...  Passed
Test #2: USB source contains resolution ... Passed
Test #3: CSI source contains nvarguscamerasrc ... Passed
Test #4: Unknown type throws ... Passed
4 tests passed, 0 failed
```

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/detection.h src/pipeline/camera_source.h \
        src/pipeline/camera_source.cpp tests/test_camera_source.cpp
git commit -m "feat: Detection struct and camera source factory"
```

---

## Task 3: Detection Probe

**Files:**
- Create: `src/pipeline/probe.h`
- Create: `src/pipeline/probe.cpp`

> Requires DeepStream headers (`nvdsmeta.h`, `gstnvdsmeta.h`). Builds and tested inside Docker only.

- [ ] **Step 1: Create src/pipeline/probe.h**

```cpp
#pragma once
#include "detection.h"
#include <functional>
#include <gst/gst.h>
#include <vector>

using DetectionCallback = std::function<void(const std::vector<Detection>&)>;

GstPadProbeReturn detection_probe_cb(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);
```

- [ ] **Step 2: Create src/pipeline/probe.cpp**

```cpp
#include "probe.h"
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

GstPadProbeReturn detection_probe_cb(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
    auto* cb = static_cast<DetectionCallback*>(user_data);

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch) return GST_PAD_PROBE_OK;

    std::vector<Detection> detections;

    for (auto* lf = batch->frame_meta_list; lf; lf = lf->next) {
        auto* frame = static_cast<NvDsFrameMeta*>(lf->data);
        for (auto* lo = frame->obj_meta_list; lo; lo = lo->next) {
            auto* obj = static_cast<NvDsObjectMeta*>(lo->data);
            const auto& r = obj->rect_params;
            detections.push_back({
                .camera_id    = static_cast<int>(frame->source_id),
                .class_id     = static_cast<int>(obj->class_id),
                .confidence   = obj->confidence,
                .left         = r.left,
                .top          = r.top,
                .width        = r.width,
                .height       = r.height,
                .timestamp_ns = frame->buf_pts,
            });
        }
    }

    if (!detections.empty()) (*cb)(detections);
    return GST_PAD_PROBE_OK;
}
```

- [ ] **Step 3: Build heimdall_pipeline inside Docker**

```bash
cmake --build build --target heimdall_pipeline
```

Expected: compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/probe.h src/pipeline/probe.cpp
git commit -m "feat: GStreamer pad probe extracts NvDsObjectMeta into Detection"
```

---

## Task 4: RFDetr Bbox Parser + nvinfer Config

**Files:**
- Modify: `src/models/bbox_parser/CMakeLists.txt` (add source file)
- Create: `src/models/bbox_parser/rfdetr_parser.cpp`
- Create: `config/infer_rfdetr.txt`

> DeepStream's `nvinfer` cannot natively parse DETR-style output (raw logits + normalized box coordinates). This shared library is loaded by nvinfer at runtime via `custom-lib-path`.

> **Before building**, verify your `.engine` output layer names:
> ```bash
> python3 -c "
> import tensorrt as trt
> r = trt.Runtime(trt.Logger(trt.Logger.WARNING))
> eng = r.deserialize_cuda_engine(open('models/rfdetr.engine','rb').read())
> [print(eng.get_binding_name(i), eng.get_binding_shape(i)) for i in range(eng.num_bindings)]
> "
> ```
> Update the `LOGITS_LAYER` / `BOXES_LAYER` string constants in `rfdetr_parser.cpp` if your export uses different names.

- [ ] **Step 1: Update src/models/bbox_parser/CMakeLists.txt**

```cmake
find_package(CUDA REQUIRED)

add_library(rfdetr_parser SHARED rfdetr_parser.cpp)
target_include_directories(rfdetr_parser PRIVATE
    "${DS_ROOT}/sources/includes"
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(rfdetr_parser ${CUDA_LIBRARIES})
```

- [ ] **Step 2: Create src/models/bbox_parser/rfdetr_parser.cpp**

```cpp
// Custom DeepStream bbox parser for RFDetr DETR-style TensorRT output.
// Expected output layers:
//   pred_logits  [num_queries, num_classes]  — raw logits (softmax applied here)
//   pred_boxes   [num_queries, 4]            — (cx, cy, w, h) normalized 0..1

#include "nvdsinfer_custom_impl.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

static constexpr const char* LOGITS_LAYER = "pred_logits";
static constexpr const char* BOXES_LAYER  = "pred_boxes";

extern "C" bool NvDsInferParseCustomRFDetr(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo            const& networkInfo,
    NvDsInferParseDetectionParams   const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList
) {
    const float* logits     = nullptr;
    const float* boxes      = nullptr;
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

    if (!logits || !boxes) return false;

    for (int q = 0; q < num_queries; ++q) {
        const float* ql = logits + q * num_classes;

        // Numerically stable softmax
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
```

- [ ] **Step 3: Create config/infer_rfdetr.txt**

```ini
# config/infer_rfdetr.txt
# Replace ENGINE_PATH, LABELS_PATH, NUM_CLASSES, PARSER_SO_PATH before use.

[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=ENGINE_PATH
labelfile-path=LABELS_PATH
batch-size=2
network-mode=1
num-detected-classes=NUM_CLASSES
interval=0
gie-unique-id=1
network-type=0
parse-bbox-func-name=NvDsInferParseCustomRFDetr
custom-lib-path=PARSER_SO_PATH

[class-attrs-all]
threshold=0.3
nms-iou-threshold=0.45
```

> `network-mode=1` = INT8. Set to `0` (FP32) for first-run debugging. `batch-size` must equal number of cameras. `PARSER_SO_PATH` = `/app/build/librfdetr_parser.so`.

- [ ] **Step 4: Build the parser inside Docker**

```bash
cmake --build build --target rfdetr_parser
ls build/src/models/bbox_parser/librfdetr_parser.so
```

Expected: file exists, no build errors.

- [ ] **Step 5: Commit**

```bash
git add src/models/bbox_parser/CMakeLists.txt \
        src/models/bbox_parser/rfdetr_parser.cpp \
        config/infer_rfdetr.txt
git commit -m "feat: RFDetr custom bbox parser and nvinfer config template"
```

---

## Task 5: Pipeline Builder

**Files:**
- Create: `src/pipeline/pipeline.h`
- Create: `src/pipeline/pipeline.cpp`
- Create: `src/main.cpp`

> Requires Jetson hardware. Build and run inside Docker on device.

- [ ] **Step 1: Create src/pipeline/pipeline.h**

```cpp
#pragma once
#include "camera_source.h"
#include "probe.h"
#include <gst/gst.h>
#include <string>
#include <vector>

class DeepStreamPipeline {
public:
    DeepStreamPipeline(
        std::vector<CameraConfig> cameras,
        std::string               infer_config_path,
        DetectionCallback         on_detection
    );
    ~DeepStreamPipeline();

    void build();
    void run();    // blocks until stop() or error
    void stop();

private:
    std::vector<CameraConfig> cameras_;
    std::string               infer_config_path_;
    DetectionCallback         on_detection_;
    GstElement*               pipeline_ = nullptr;
    GMainLoop*                loop_     = nullptr;

    static gboolean bus_cb(GstBus*, GstMessage*, gpointer);
};
```

- [ ] **Step 2: Create src/pipeline/pipeline.cpp**

```cpp
#include "pipeline.h"
#include <stdexcept>

DeepStreamPipeline::DeepStreamPipeline(
    std::vector<CameraConfig> cameras,
    std::string               infer_config_path,
    DetectionCallback         on_detection
) : cameras_(std::move(cameras)),
    infer_config_path_(std::move(infer_config_path)),
    on_detection_(std::move(on_detection)) {}

DeepStreamPipeline::~DeepStreamPipeline() { stop(); }

void DeepStreamPipeline::build() {
    gst_init(nullptr, nullptr);
    pipeline_ = gst_pipeline_new("heimdall");

    GstElement* mux = gst_element_factory_make("nvstreammux", "mux");
    g_object_set(mux,
        "width",                640,
        "height",               480,
        "batch-size",           static_cast<gint>(cameras_.size()),
        "batched-push-timeout", 4000000,
        "live-source",          TRUE,
        nullptr);
    gst_bin_add(GST_BIN(pipeline_), mux);

    for (int i = 0; i < static_cast<int>(cameras_.size()); ++i) {
        GError* err = nullptr;
        GstElement* src = gst_parse_bin_from_description(
            build_source_description(cameras_[i]).c_str(), TRUE, &err);
        if (!src) throw std::runtime_error(err->message);

        gst_element_set_name(src, ("src_" + std::to_string(i)).c_str());
        gst_bin_add(GST_BIN(pipeline_), src);

        GstPad* src_pad  = gst_element_get_static_pad(src, "src");
        GstPad* sink_pad = gst_element_get_request_pad(mux,
            ("sink_" + std::to_string(i)).c_str());
        gst_pad_link(src_pad, sink_pad);
        gst_object_unref(src_pad);
        gst_object_unref(sink_pad);
    }

    GstElement* infer = gst_element_factory_make("nvinfer", "infer");
    g_object_set(infer, "config-file-path", infer_config_path_.c_str(), nullptr);
    gst_bin_add(GST_BIN(pipeline_), infer);

    GstElement* sink = gst_element_factory_make("fakesink", "sink");
    g_object_set(sink, "sync", FALSE, nullptr);
    gst_bin_add(GST_BIN(pipeline_), sink);

    gst_element_link(mux, infer);
    gst_element_link(infer, sink);

    GstPad* infer_src = gst_element_get_static_pad(infer, "src");
    gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER,
        detection_probe_cb, &on_detection_, nullptr);
    gst_object_unref(infer_src);

    GstBus* bus = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus, bus_cb, this);
    gst_object_unref(bus);
}

gboolean DeepStreamPipeline::bus_cb(GstBus*, GstMessage* msg, gpointer data) {
    auto* self = static_cast<DeepStreamPipeline*>(data);
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError* err; gchar* dbg;
            gst_message_parse_error(msg, &err, &dbg);
            g_printerr("Pipeline error: %s\n%s\n", err->message, dbg ? dbg : "");
            g_error_free(err); g_free(dbg);
            if (self->loop_) g_main_loop_quit(self->loop_);
            break;
        }
        case GST_MESSAGE_EOS:
            if (self->loop_) g_main_loop_quit(self->loop_);
            break;
        default: break;
    }
    return TRUE;
}

void DeepStreamPipeline::run() {
    if (!pipeline_) build();
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    loop_ = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(loop_);
}

void DeepStreamPipeline::stop() {
    if (loop_)     { g_main_loop_quit(loop_); g_main_loop_unref(loop_); loop_ = nullptr; }
    if (pipeline_) { gst_element_set_state(pipeline_, GST_STATE_NULL);
                     gst_object_unref(pipeline_); pipeline_ = nullptr; }
}
```

- [ ] **Step 3: Create src/main.cpp**

```cpp
#include "pipeline/camera_source.h"
#include "pipeline/detection.h"
#include "pipeline/pipeline.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <vector>

static DeepStreamPipeline* g_pipeline = nullptr;
static std::atomic<int>    g_frames{0};
static auto                g_start = std::chrono::steady_clock::now();

static void on_detection(const std::vector<Detection>& dets) {
    ++g_frames;
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g_start).count();

    for (const auto& d : dets)
        std::printf("cam=%d class=%d conf=%.2f box=(%.0f,%.0f,%.0fx%.0f)\n",
            d.camera_id, d.class_id, d.confidence,
            d.left, d.top, d.width, d.height);

    std::printf("  FPS: %.1f  frames=%d\n",
        g_frames.load() / elapsed, g_frames.load());
}

static void shutdown(int) { if (g_pipeline) g_pipeline->stop(); }

int main() {
    std::vector<CameraConfig> cameras{
        {0, CameraType::USB, "/dev/video0"},
    };

    DeepStreamPipeline pipeline(cameras, "config/infer_rfdetr.txt", on_detection);
    g_pipeline = &pipeline;
    std::signal(SIGINT, shutdown);

    g_start = std::chrono::steady_clock::now();
    pipeline.run();
}
```

- [ ] **Step 4: Build full project inside Docker**

```bash
cmake --build build --parallel $(nproc)
ls build/heimdall
```

Expected: `build/heimdall` exists, no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/pipeline.h src/pipeline/pipeline.cpp src/main.cpp
git commit -m "feat: DeepStreamPipeline class and validation entry point"
```

---

## Task 6: Jetson Integration Validation

**Prerequisites:**
1. `models/rfdetr.engine` — INT8 TensorRT engine on Jetson
2. `models/labels.txt` — one class name per line
3. `config/infer_rfdetr.txt` — filled in (ENGINE_PATH, LABELS_PATH, NUM_CLASSES, PARSER_SO_PATH)
4. At least one USB camera at `/dev/video0`

- [ ] **Step 1: Fill in config/infer_rfdetr.txt**

Replace placeholders:
```
ENGINE_PATH   → /app/models/rfdetr.engine
LABELS_PATH   → /app/models/labels.txt
NUM_CLASSES   → (count lines in labels.txt)
PARSER_SO_PATH → /app/build/src/models/bbox_parser/librfdetr_parser.so
```

- [ ] **Step 2: Build and run inside Docker on Jetson**

```bash
cd docker
docker compose up --build
```

Expected output (per frame):
```
cam=0 class=0 conf=0.87 box=(120,80,64x48)
  FPS: 52.3  frames=42
```

- [ ] **Step 3: Verify spike criteria**

Both must be true after 10 seconds of warmup:
- [ ] Detections print with valid class IDs and bounding boxes
- [ ] FPS ≥ 50 per camera

**If FPS < 50:**
1. Confirm `network-mode=1` (INT8) matches engine precision
2. Check camera outputs MJPEG: `v4l2-ctl --list-formats-ext -d /dev/video0`
3. Lower `batched-push-timeout` (e.g. `2000000`) to reduce mux latency

- [ ] **Step 4: Commit**

```bash
git add config/infer_rfdetr.txt
git commit -m "feat: fill nvinfer config for Jetson validation"
```

---

## Spike Complete — Criteria

| Check | Condition |
|-------|-----------|
| Unit tests | `ctest` passes 4/4 in Docker |
| Docker build | `docker compose build` succeeds |
| Live detections | Valid class + bbox printed each frame |
| FPS | ≥50 per camera after 10s warmup |

**Milestone 2** adds ground ray pose estimation on top of the `Detection` structs from this pipeline.
