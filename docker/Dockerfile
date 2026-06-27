# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/deepstream-l4t:7.1-triton-multiarch

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    git \
    ccache \
    libapriltag-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only build config — layer cached until deps change
COPY CMakeLists.txt .
COPY src/models/bbox_parser/CMakeLists.txt src/models/bbox_parser/CMakeLists.txt
COPY proto/ proto/
COPY tests/CMakeLists.txt tests/CMakeLists.txt
# Stub source files so cmake configure succeeds before real sources are copied
RUN mkdir -p src/tracker src/pose src/comm src/pipeline src/app \
             src/apriltag src/models/bbox_parser src/config tools tests && \
    touch src/tracker/kalman.cpp src/tracker/jpda.cpp src/tracker/tracker.cpp \
          src/pose/pose_estimator.cpp \
          src/comm/comm_layer.cpp \
          src/pipeline/camera_source.cpp src/pipeline/probe.cpp src/pipeline/pipeline.cpp \
          src/app/heimdall_app.cpp src/app/detection_processor.cpp \
          src/app/output_jsonl.cpp src/app/recording.cpp src/app/replay_main.cpp \
          src/main.cpp src/config/app_config_loader.cpp \
          src/apriltag/tag_layout.cpp src/apriltag/apriltag_detector.cpp \
          src/models/bbox_parser/rfdetr_parser.cpp \
          src/models/bbox_parser/yolo_parser.cpp \
          src/config/camera_config_loader.cpp \
          tools/mock_robot.cpp \
          tests/test_camera_source.cpp tests/test_pipeline_chain.cpp \
          tests/test_yolo_parser.cpp \
          tests/test_kalman.cpp tests/test_pose_estimator.cpp \
          tests/test_comm_layer.cpp tests/test_jpda.cpp tests/test_tracker.cpp \
          tests/test_pose_buffer.cpp
RUN PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache

# Build third-party deps — cached until CMakeLists.txt changes
RUN --mount=type=cache,target=/root/.ccache \
    cmake --build build --parallel $(nproc) \
    --target libzmq libprotobuf protoc Catch2WithMain

# Build our code — deps already compiled, only our files recompile on source changes
COPY . .
RUN --mount=type=cache,target=/root/.ccache \
    cmake --build build --parallel $(nproc)

CMD ["/app/build/heimdall"]
