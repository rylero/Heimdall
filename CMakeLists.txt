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
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${DS_INCLUDES}
    ${GST_INCLUDE_DIRS}
)
target_link_libraries(heimdall_pipeline
    PUBLIC
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
