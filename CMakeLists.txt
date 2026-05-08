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
# libzmq — ZeroMQ C library
# ---------------------------------------------------------------------------
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(
    zeromq
    GIT_REPOSITORY https://github.com/zeromq/libzmq.git
    GIT_TAG        v4.3.5
    GIT_SHALLOW    TRUE
)
set(ZMQ_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(WITH_PERF_TOOL    OFF CACHE BOOL "" FORCE)
set(WITH_DOCS         OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(zeromq)

# ---------------------------------------------------------------------------
# cppzmq — header-only C++ bindings for libzmq
# ---------------------------------------------------------------------------
FetchContent_Declare(
    cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG        v4.10.0
    GIT_SHALLOW    TRUE
)
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cppzmq)

# ---------------------------------------------------------------------------
# Protobuf — message serialization
# ---------------------------------------------------------------------------
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG        v26.1
    GIT_SHALLOW    TRUE
)
set(protobuf_BUILD_TESTS         OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES      OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL             OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(protobuf)
# protobuf_generate() lives in cmake/protobuf-generate.cmake; must be included explicitly
# because FetchContent runs the subproject in its own scope (functions don't propagate up)
include(${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)

# ---------------------------------------------------------------------------
# Tracker library — no GStreamer/DeepStream dependency, unit-testable anywhere
# ---------------------------------------------------------------------------
add_library(heimdall_tracker
    src/tracker/kalman.cpp
    src/tracker/jpda.cpp
    src/tracker/tracker.cpp
)
target_include_directories(heimdall_tracker PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ---------------------------------------------------------------------------
# Pose estimation library — no hardware dependency, unit-testable anywhere
# ---------------------------------------------------------------------------
add_library(heimdall_pose
    src/pose/pose_estimator.cpp
)
target_include_directories(heimdall_pose PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ---------------------------------------------------------------------------
# Protobuf generated sources
# ---------------------------------------------------------------------------
add_library(heimdall_proto
    proto/heimdall.proto
)
target_include_directories(heimdall_proto PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/proto
)
target_link_libraries(heimdall_proto PUBLIC libprotobuf)
protobuf_generate(
    TARGET  heimdall_proto
    PROTOS  ${CMAKE_CURRENT_SOURCE_DIR}/proto/heimdall.proto
    LANGUAGE cpp
)

# ---------------------------------------------------------------------------
# Comm library — ZeroMQ + protobuf, no hardware dependency
# ---------------------------------------------------------------------------
add_library(heimdall_comm
    src/comm/comm_layer.cpp
)
target_include_directories(heimdall_comm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(heimdall_comm
    PUBLIC  heimdall_proto heimdall_tracker heimdall_pose cppzmq
    PRIVATE libzmq
)

# ---------------------------------------------------------------------------
# Mock robot — standalone tool, simulates RoboRIO ZeroMQ interface
# ---------------------------------------------------------------------------
add_executable(mock_robot tools/mock_robot.cpp)
target_include_directories(mock_robot PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(mock_robot PRIVATE heimdall_comm libzmq cppzmq)

# ---------------------------------------------------------------------------
# GStreamer (available inside DeepStream Docker image)
# ---------------------------------------------------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GST      gstreamer-1.0)
    pkg_check_modules(GST_RTSP gst-rtsp-server-1.0)
endif()

# ---------------------------------------------------------------------------
# DeepStream paths
# ---------------------------------------------------------------------------
set(DS_ROOT "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream SDK root")
set(DS_INCLUDES "${DS_ROOT}/sources/includes")
set(DS_LIB_DIR  "${DS_ROOT}/lib")

find_library(NVDS_META_LIB     nvds_meta     PATHS ${DS_LIB_DIR})
find_library(NVDSGST_META_LIB  nvdsgst_meta  PATHS ${DS_LIB_DIR})

# ---------------------------------------------------------------------------
# Pipeline library (only built when GStreamer/DeepStream are available)
# ---------------------------------------------------------------------------
if(GST_FOUND AND GST_RTSP_FOUND AND NVDS_META_LIB AND NVDSGST_META_LIB)
    add_library(heimdall_pipeline
        src/pipeline/camera_source.cpp
        src/pipeline/probe.cpp
        src/pipeline/pipeline.cpp
    )
    target_include_directories(heimdall_pipeline PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${DS_INCLUDES}
        ${GST_INCLUDE_DIRS}
        ${GST_RTSP_INCLUDE_DIRS}
    )
    target_link_libraries(heimdall_pipeline
        PUBLIC
            ${GST_LIBRARIES}
            ${GST_RTSP_LIBRARIES}
            ${NVDS_META_LIB}
            ${NVDSGST_META_LIB}
            heimdall_tracker
    )

    # Main executable — full integrated app
    add_executable(heimdall
        src/main.cpp
        src/app/heimdall_app.cpp
    )
    target_include_directories(heimdall PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(heimdall
        heimdall_pipeline
        heimdall_comm
    )

    # RFDetr bbox parser shared library
    add_subdirectory(src/models/bbox_parser)
else()
    message(STATUS "GStreamer/DeepStream not found — skipping pipeline, main executable, and bbox_parser (tracker tests still build).")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
