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
# GStreamer (available inside DeepStream Docker image)
# ---------------------------------------------------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GST gstreamer-1.0)
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
if(GST_FOUND AND NVDS_META_LIB AND NVDSGST_META_LIB)
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
            heimdall_tracker
    )

    # Main executable (on-Jetson validation)
    add_executable(heimdall src/main.cpp)
    target_link_libraries(heimdall heimdall_pipeline)

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
