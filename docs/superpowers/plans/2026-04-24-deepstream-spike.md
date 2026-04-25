# Heimdall Milestone 1: DeepStream Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multi-camera DeepStream pipeline that loads an RFDetr INT8 TensorRT `.engine` file and outputs raw `Detection` objects at ≥50 FPS per camera.

**Architecture:** GStreamer pipeline (camera sources → nvstreammux → nvinfer) with a pad probe extracting NvDsObjectMeta into Python `Detection` objects. A custom C++ bbox parser handles RFDetr's DETR-style output (pred_logits + pred_boxes). All wired together in a `DeepStreamPipeline` class and containerized with Docker.

**Tech Stack:** Python 3.10+, NVIDIA DeepStream 7.x, TensorRT, pyds (DeepStream Python bindings), GStreamer (gi.repository), pytest + unittest.mock, CMake, Docker

---

## Milestone Roadmap (future plans)

| Milestone | Description |
|-----------|-------------|
| **1 (this plan)** | DeepStream multi-camera detection spike |
| 2 | Ground ray pose estimation (field-relative coordinates) |
| 3 | JPDA object tracker (global map, 80–100 objects) |
| 4 | ZeroMQ push-pull communication + Protobuf schema |
| 5 | Web interface (config, model upload, live preview) |
| 6 | Integration, tuning, and Docker deployment |

---

## File Structure

```
heimdall/
  docker/
    Dockerfile                   # DeepStream L4T container
    docker-compose.yml           # Service + volume mounts
  heimdall/
    pipeline/
      __init__.py
      camera.py                  # CameraConfig + GStreamer source string builder
      probe.py                   # Detection dataclass + pyds metadata extraction
      pipeline.py                # DeepStreamPipeline: builds + runs GStreamer graph
    models/
      bbox_parser/
        rfdetr_parser.cpp        # Custom NvDsInferParseCustom for DETR output
        CMakeLists.txt
  config/
    infer_rfdetr.txt             # nvinfer config template (engine path is a placeholder)
  tests/
    pipeline/
      __init__.py
      test_camera.py             # Pure Python — no hardware needed
      test_probe.py              # mocked pyds
  scripts/
    run_spike.py                 # On-Jetson manual validation + FPS counter
  models/
    .gitkeep                     # Drop .engine + labels.txt here
  pyproject.toml
```

> **Note on testing:** `pyds` and `gi.repository.Gst` only exist on Jetson hardware. Unit tests mock both. Tasks 1–4 run on any machine. Task 5 (pipeline builder) and Task 6 (validation) require the Jetson.

---

## Task 1: Project Scaffold + Docker

**Files:**
- Create: `pyproject.toml`
- Create: `heimdall/__init__.py`
- Create: `heimdall/pipeline/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/pipeline/__init__.py`
- Create: `models/.gitkeep`
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "heimdall"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-mock>=3"]

[tool.setuptools.packages.find]
where = ["."]
include = ["heimdall*"]
```

- [ ] **Step 2: Create empty `__init__.py` files**

```bash
touch heimdall/__init__.py
touch heimdall/pipeline/__init__.py
touch tests/__init__.py
touch tests/pipeline/__init__.py
touch models/.gitkeep
```

- [ ] **Step 3: Create Dockerfile**

```dockerfile
FROM nvcr.io/nvidia/deepstream-l4t:7.0-triton

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-gi \
    python3-gst-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    libgstreamer1.0-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pyds ships inside the DeepStream image at this path; install it
RUN pip3 install /opt/nvidia/deepstream/deepstream/lib/pyds-*.whl

WORKDIR /app
COPY . .

RUN pip3 install -e ".[dev]"

# Build RFDetr bbox parser shared library
RUN cmake -S heimdall/models/bbox_parser -B build/bbox_parser \
    && cmake --build build/bbox_parser \
    && cp build/bbox_parser/librfdetr_parser.so heimdall/models/bbox_parser/rfdetr_parser.so

CMD ["python3", "scripts/run_spike.py"]
```

- [ ] **Step 4: Create docker-compose.yml**

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
      - ../models:/app/models       # drop engine files here without rebuild
      - ../config:/app/config
    devices:
      - /dev/video0:/dev/video0     # USB camera 0
      - /dev/video1:/dev/video1     # USB camera 1
    network_mode: host
```

- [ ] **Step 5: Verify Python package installs**

```bash
pip install -e ".[dev]"
python -c "import heimdall; print('ok')"
```

Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml heimdall/ tests/ models/ docker/ config/
git commit -m "feat: project scaffold and Docker setup"
```

---

## Task 2: Camera Source Factory

**Files:**
- Create: `heimdall/pipeline/camera.py`
- Create: `tests/pipeline/test_camera.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_camera.py
import pytest
from heimdall.pipeline.camera import CameraConfig, CameraType, build_source_bin


def test_usb_source_contains_v4l2src():
    config = CameraConfig(id=0, type=CameraType.USB, device="/dev/video0")
    result = build_source_bin(config)
    assert "v4l2src" in result
    assert "device=/dev/video0" in result


def test_usb_source_contains_resolution():
    config = CameraConfig(id=0, type=CameraType.USB, device="/dev/video0", width=640, height=480)
    result = build_source_bin(config)
    assert "width=640" in result
    assert "height=480" in result


def test_csi_source_contains_nvargus():
    config = CameraConfig(id=1, type=CameraType.CSI, device="0")
    result = build_source_bin(config)
    assert "nvarguscamerasrc" in result
    assert "sensor-id=0" in result


def test_unknown_type_raises():
    config = CameraConfig(id=0, type="bad", device="/dev/video0")  # type: ignore
    with pytest.raises(ValueError, match="Unknown camera type"):
        build_source_bin(config)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/pipeline/test_camera.py -v
```

Expected: `ModuleNotFoundError: No module named 'heimdall.pipeline.camera'`

- [ ] **Step 3: Implement camera.py**

```python
# heimdall/pipeline/camera.py
from dataclasses import dataclass, field
from enum import Enum


class CameraType(Enum):
    USB = "usb"
    CSI = "csi"


@dataclass
class CameraConfig:
    id: int
    type: CameraType
    device: str          # "/dev/video0" for USB, "0" for CSI sensor index
    width: int = 640
    height: int = 480
    fps: int = 60


def build_source_bin(config: CameraConfig) -> str:
    """Return a GStreamer element description string for a single camera source."""
    if config.type == CameraType.USB:
        return (
            f"v4l2src device={config.device} ! "
            f"image/jpeg,width={config.width},height={config.height},framerate={config.fps}/1 ! "
            f"jpegdec ! nvvidconv ! "
            f"video/x-raw(memory:NVMM),format=NV12"
        )
    if config.type == CameraType.CSI:
        return (
            f"nvarguscamerasrc sensor-id={config.device} ! "
            f"video/x-raw(memory:NVMM),width={config.width},height={config.height},"
            f"format=NV12,framerate={config.fps}/1"
        )
    raise ValueError(f"Unknown camera type: {config.type}")
```

> **Note:** USB cameras that output MJPEG are the common fast path on Jetson. If a camera only supports raw YUV, replace `image/jpeg ... jpegdec` with `video/x-raw,format=YUY2 ! videoconvert`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_camera.py -v
```

Expected:
```
PASSED tests/pipeline/test_camera.py::test_usb_source_contains_v4l2src
PASSED tests/pipeline/test_camera.py::test_usb_source_contains_resolution
PASSED tests/pipeline/test_camera.py::test_csi_source_contains_nvargus
PASSED tests/pipeline/test_camera.py::test_unknown_type_raises
4 passed in 0.XXs
```

- [ ] **Step 5: Commit**

```bash
git add heimdall/pipeline/camera.py tests/pipeline/test_camera.py
git commit -m "feat: camera source factory for USB and CSI"
```

---

## Task 3: Detection Dataclass + pyds Probe

**Files:**
- Create: `heimdall/pipeline/probe.py`
- Create: `tests/pipeline/test_probe.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_probe.py
from unittest.mock import MagicMock, patch
from heimdall.pipeline.probe import Detection, extract_detections


def _make_mock_obj(class_id, confidence, left, top, width, height, next_obj=None):
    obj = MagicMock()
    obj.class_id = class_id
    obj.confidence = confidence
    obj.rect_params.left = left
    obj.rect_params.top = top
    obj.rect_params.width = width
    obj.rect_params.height = height
    obj.next = next_obj
    return obj


def _make_mock_frame(source_id, buf_pts, obj_meta, next_frame=None):
    frame = MagicMock()
    frame.source_id = source_id
    frame.buf_pts = buf_pts
    frame.obj_meta_list = obj_meta
    frame.next = next_frame
    return frame


def _make_mock_batch(frame):
    batch = MagicMock()
    batch.frame_meta_list = frame
    return batch


def test_extract_single_detection():
    obj = _make_mock_obj(2, 0.91, 100.0, 50.0, 80.0, 60.0)
    frame = _make_mock_frame(0, 1_000_000, obj)
    batch = _make_mock_batch(frame)
    buf = MagicMock()

    with patch.dict("sys.modules", {"pyds": MagicMock()}):
        import pyds
        pyds.gst_buffer_get_nvds_batch_meta.return_value = batch
        pyds.NvDsFrameMeta.cast.side_effect = lambda x: x
        pyds.NvDsObjectMeta.cast.side_effect = lambda x: x

        result = extract_detections(buf)

    assert len(result) == 1
    d = result[0]
    assert d.camera_id == 0
    assert d.class_id == 2
    assert d.confidence == 0.91
    assert d.left == 100.0
    assert d.top == 50.0
    assert d.width == 80.0
    assert d.height == 60.0
    assert d.timestamp_ns == 1_000_000


def test_extract_no_detections_returns_empty():
    frame = _make_mock_frame(0, 0, None)
    batch = _make_mock_batch(frame)
    buf = MagicMock()

    with patch.dict("sys.modules", {"pyds": MagicMock()}):
        import pyds
        pyds.gst_buffer_get_nvds_batch_meta.return_value = batch
        pyds.NvDsFrameMeta.cast.side_effect = lambda x: x
        pyds.NvDsObjectMeta.cast.side_effect = lambda x: x

        result = extract_detections(buf)

    assert result == []


def test_extract_multi_camera():
    obj0 = _make_mock_obj(0, 0.80, 10, 10, 30, 30)
    obj1 = _make_mock_obj(1, 0.75, 200, 100, 50, 40)
    frame1 = _make_mock_frame(1, 500, obj1)
    frame0 = _make_mock_frame(0, 500, obj0, next_frame=frame1)
    batch = _make_mock_batch(frame0)
    buf = MagicMock()

    with patch.dict("sys.modules", {"pyds": MagicMock()}):
        import pyds
        pyds.gst_buffer_get_nvds_batch_meta.return_value = batch
        pyds.NvDsFrameMeta.cast.side_effect = lambda x: x
        pyds.NvDsObjectMeta.cast.side_effect = lambda x: x

        result = extract_detections(buf)

    assert len(result) == 2
    cam_ids = {d.camera_id for d in result}
    assert cam_ids == {0, 1}
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/pipeline/test_probe.py -v
```

Expected: `ModuleNotFoundError: No module named 'heimdall.pipeline.probe'`

- [ ] **Step 3: Implement probe.py**

```python
# heimdall/pipeline/probe.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Detection:
    camera_id: int
    class_id: int
    confidence: float
    left: float       # pixels, 640×480 space
    top: float
    width: float
    height: float
    timestamp_ns: int


def extract_detections(gst_buffer) -> list[Detection]:
    """Extract all NvDsObjectMeta from a GStreamer buffer into Detection objects."""
    import pyds

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    detections: list[Detection] = []

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            r = obj.rect_params
            detections.append(Detection(
                camera_id=frame_meta.source_id,
                class_id=obj.class_id,
                confidence=obj.confidence,
                left=r.left,
                top=r.top,
                width=r.width,
                height=r.height,
                timestamp_ns=frame_meta.buf_pts,
            ))
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return detections
```

> **Note:** pyds iterators raise `StopIteration` at end-of-list in some DeepStream versions. The `try/except` guards against this.

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_probe.py -v
```

Expected:
```
PASSED tests/pipeline/test_probe.py::test_extract_single_detection
PASSED tests/pipeline/test_probe.py::test_extract_no_detections_returns_empty
PASSED tests/pipeline/test_probe.py::test_extract_multi_camera
3 passed in 0.XXs
```

- [ ] **Step 5: Commit**

```bash
git add heimdall/pipeline/probe.py tests/pipeline/test_probe.py
git commit -m "feat: Detection dataclass and pyds metadata extraction probe"
```

---

## Task 4: RFDetr Bbox Parser (C++) + nvinfer Config

**Files:**
- Create: `heimdall/models/bbox_parser/rfdetr_parser.cpp`
- Create: `heimdall/models/bbox_parser/CMakeLists.txt`
- Create: `config/infer_rfdetr.txt`

DeepStream's `nvinfer` cannot natively parse DETR-style output (raw logits + box coordinates). This C++ shared library is loaded by nvinfer to convert RFDetr output into `NvDsInferParseObjectInfo` structs.

> **Before building:** Confirm your exported `.engine` file's output layer names. Run:
> ```bash
> python3 -c "
> import tensorrt as trt
> r = trt.Runtime(trt.Logger(trt.Logger.WARNING))
> with open('models/rfdetr.engine','rb') as f:
>     eng = r.deserialize_cuda_engine(f.read())
> for i in range(eng.num_bindings):
>     print(eng.get_binding_name(i), eng.get_binding_shape(i))
> "
> ```
> Update `rfdetr_parser.cpp` layer name strings to match if different from `pred_logits` / `pred_boxes`.

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.16)
project(rfdetr_parser)

find_package(CUDA REQUIRED)

set(DS_INCLUDES "/opt/nvidia/deepstream/deepstream/sources/includes"
    CACHE PATH "DeepStream SDK include directory")

include_directories(${DS_INCLUDES} ${CUDA_INCLUDE_DIRS})

add_library(rfdetr_parser SHARED rfdetr_parser.cpp)
target_link_libraries(rfdetr_parser ${CUDA_LIBRARIES})
set_target_properties(rfdetr_parser PROPERTIES
    OUTPUT_NAME "rfdetr_parser"
    PREFIX "lib"
)
```

- [ ] **Step 2: Create rfdetr_parser.cpp**

```cpp
// heimdall/models/bbox_parser/rfdetr_parser.cpp
//
// Custom DeepStream bbox parser for RFDetr DETR-style TensorRT output.
// Output layers expected:
//   pred_logits  [num_queries, num_classes]  — raw logits, softmax applied here
//   pred_boxes   [num_queries, 4]            — (cx, cy, w, h) normalized 0..1
//
// Update layer name strings below if your export produces different names.

#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "nvdsinfer_custom_impl.h"

static const char* LOGITS_LAYER = "pred_logits";
static const char* BOXES_LAYER  = "pred_boxes";

extern "C" bool NvDsInferParseCustomRFDetr(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo  const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList
) {
    const float* logits = nullptr;
    const float* boxes  = nullptr;
    int num_queries = 0;
    int num_classes = 0;

    for (const auto& layer : outputLayersInfo) {
        if (std::string(layer.layerName) == LOGITS_LAYER) {
            logits      = static_cast<const float*>(layer.buffer);
            num_queries = layer.inferDims.d[0];
            num_classes = layer.inferDims.d[1];
        } else if (std::string(layer.layerName) == BOXES_LAYER) {
            boxes = static_cast<const float*>(layer.buffer);
        }
    }

    if (!logits || !boxes) {
        return false;  // layer names didn't match — check engine bindings
    }

    for (int q = 0; q < num_queries; ++q) {
        const float* q_logits = logits + q * num_classes;

        // softmax over classes
        float max_logit = q_logits[0];
        for (int c = 1; c < num_classes; ++c)
            if (q_logits[c] > max_logit) max_logit = q_logits[c];

        float sum_exp = 0.f;
        for (int c = 0; c < num_classes; ++c)
            sum_exp += std::exp(q_logits[c] - max_logit);

        float best_score = -1.f;
        int   best_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = std::exp(q_logits[c] - max_logit) / sum_exp;
            if (score > best_score) { best_score = score; best_class = c; }
        }

        float threshold = (best_class < (int)detectionParams.perClassThreshold.size())
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

- [ ] **Step 3: Create infer_rfdetr.txt**

```ini
# config/infer_rfdetr.txt
# Replace ENGINE_PATH, LABELS_PATH, NUM_CLASSES, and PARSER_SO_PATH before use.

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

> `network-mode=1` = INT8. Set to `0` (FP32) for first-run debugging if INT8 engine has issues. `batch-size` must equal number of cameras.

- [ ] **Step 4: Build the parser (on Jetson or in Docker)**

```bash
cmake -S heimdall/models/bbox_parser -B build/bbox_parser
cmake --build build/bbox_parser
ls build/bbox_parser/librfdetr_parser.so
```

Expected: `build/bbox_parser/librfdetr_parser.so` exists, no build errors.

- [ ] **Step 5: Commit**

```bash
git add heimdall/models/bbox_parser/ config/infer_rfdetr.txt
git commit -m "feat: RFDetr custom bbox parser and nvinfer config template"
```

---

## Task 5: Pipeline Builder

**Files:**
- Create: `heimdall/pipeline/pipeline.py`

> **Hardware required from here.** Runs inside the Docker container on Jetson. No further unit tests — the GStreamer runtime is not mockable at a useful level for a spike. Correctness is validated by Task 6.

- [ ] **Step 1: Create pipeline.py**

```python
# heimdall/pipeline/pipeline.py
from __future__ import annotations
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from heimdall.pipeline.camera import CameraConfig, build_source_bin
from heimdall.pipeline.probe import Detection, extract_detections
from typing import Callable


class DeepStreamPipeline:
    def __init__(
        self,
        cameras: list[CameraConfig],
        infer_config_path: str,
        on_detection: Callable[[list[Detection]], None],
    ) -> None:
        self._cameras = cameras
        self._infer_config_path = infer_config_path
        self._on_detection = on_detection
        self._pipeline: Gst.Pipeline | None = None
        self._loop: GLib.MainLoop | None = None

    def build(self) -> None:
        Gst.init(None)
        self._pipeline = Gst.Pipeline.new("heimdall")

        streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        streammux.set_property("width", 640)
        streammux.set_property("height", 480)
        streammux.set_property("batch-size", len(self._cameras))
        streammux.set_property("batched-push-timeout", 4_000_000)
        streammux.set_property("live-source", True)
        self._pipeline.add(streammux)

        for i, cam in enumerate(self._cameras):
            src_bin = Gst.parse_bin_from_description(build_source_bin(cam), True)
            src_bin.set_name(f"src_{i}")
            self._pipeline.add(src_bin)
            src_pad = src_bin.get_static_pad("src")
            sink_pad = streammux.get_request_pad(f"sink_{i}")
            src_pad.link(sink_pad)

        infer = Gst.ElementFactory.make("nvinfer", "infer")
        infer.set_property("config-file-path", self._infer_config_path)
        self._pipeline.add(infer)

        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        self._pipeline.add(sink)

        streammux.link(infer)
        infer.link(sink)

        infer_src_pad = infer.get_static_pad("src")
        infer_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._buffer_probe,
        )

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)
        bus.connect("message::eos", self._on_eos)

    def _buffer_probe(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        buf = info.get_buffer()
        detections = extract_detections(buf)
        if detections:
            self._on_detection(detections)
        return Gst.PadProbeReturn.OK

    def _on_error(self, bus: Gst.Bus, msg: Gst.Message) -> None:
        err, debug = msg.parse_error()
        print(f"[Pipeline ERROR] {err}: {debug}")
        if self._loop:
            self._loop.quit()

    def _on_eos(self, bus: Gst.Bus, msg: Gst.Message) -> None:
        print("[Pipeline] EOS received")
        if self._loop:
            self._loop.quit()

    def run(self) -> None:
        """Build and block until the pipeline stops."""
        self.build()
        self._pipeline.set_state(Gst.State.PLAYING)
        self._loop = GLib.MainLoop()
        self._loop.run()

    def stop(self) -> None:
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop:
            self._loop.quit()
```

- [ ] **Step 2: Commit**

```bash
git add heimdall/pipeline/pipeline.py
git commit -m "feat: DeepStream GStreamer pipeline builder"
```

---

## Task 6: Jetson Integration Validation

**Files:**
- Create: `scripts/run_spike.py`

**Prerequisites:**
1. `models/rfdetr.engine` — INT8 TensorRT engine on Jetson
2. `models/labels.txt` — one class name per line
3. `config/infer_rfdetr.txt` — filled in with correct paths + `NUM_CLASSES`
4. `build/bbox_parser/librfdetr_parser.so` — built in Task 4
5. At least one USB camera at `/dev/video0`

- [ ] **Step 1: Create run_spike.py**

```python
#!/usr/bin/env python3
# scripts/run_spike.py
import time
import signal
from heimdall.pipeline.camera import CameraConfig, CameraType
from heimdall.pipeline.probe import Detection
from heimdall.pipeline.pipeline import DeepStreamPipeline

CAMERAS = [
    CameraConfig(id=0, type=CameraType.USB, device="/dev/video0"),
    # CameraConfig(id=1, type=CameraType.USB, device="/dev/video1"),
]
INFER_CONFIG = "config/infer_rfdetr.txt"

frame_count = 0
start_time = time.monotonic()

def on_detection(detections: list[Detection]) -> None:
    global frame_count
    frame_count += 1

    elapsed = time.monotonic() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0.0

    for d in detections:
        print(
            f"cam={d.camera_id} class={d.class_id} conf={d.confidence:.2f} "
            f"box=({d.left:.0f},{d.top:.0f},{d.width:.0f}x{d.height:.0f}) "
            f"ts={d.timestamp_ns}"
        )
    print(f"  FPS: {fps:.1f}  total_frames={frame_count}")

pipeline = DeepStreamPipeline(CAMERAS, INFER_CONFIG, on_detection)

def _shutdown(sig, frame):
    print("\nShutting down...")
    pipeline.stop()

signal.signal(signal.SIGINT, _shutdown)
pipeline.run()
```

- [ ] **Step 2: Fill in infer_rfdetr.txt**

Edit `config/infer_rfdetr.txt`, replacing all-caps placeholders:

```
ENGINE_PATH   → /app/models/rfdetr.engine
LABELS_PATH   → /app/models/labels.txt
NUM_CLASSES   → (number of lines in labels.txt)
PARSER_SO_PATH → /app/build/bbox_parser/librfdetr_parser.so
```

- [ ] **Step 3: Run inside Docker on Jetson**

```bash
cd docker
docker compose up --build
```

Expected output (per detection frame):
```
cam=0 class=0 conf=0.87 box=(120,80,64x48) ts=123456789
  FPS: 52.3  total_frames=42
```

- [ ] **Step 4: Verify spike criteria**

Check both conditions are met:
- [ ] Detections print with valid class IDs and bounding boxes
- [ ] FPS reported ≥50 per camera after 10 seconds of warmup

If FPS < 50, check:
1. `network-mode` in infer config matches engine precision (INT8 = `1`)
2. Camera is outputting MJPEG (not raw YUY2 which is slower to decode) — run `v4l2-ctl --list-formats-ext -d /dev/video0`
3. `batched-push-timeout` — lower value reduces latency but can cause dropped frames

- [ ] **Step 5: Commit**

```bash
git add scripts/run_spike.py
git commit -m "feat: Jetson integration validation script for DeepStream spike"
```

---

## Spike Complete — Next Steps

Milestone 1 done when:
- `pytest tests/` passes on any machine (Tasks 1–3)
- Docker container builds on Jetson (Task 4)
- `run_spike.py` shows ≥50 FPS detections (Task 6)

**Milestone 2** will layer ground ray pose estimation on top of raw `Detection` objects from this pipeline.
