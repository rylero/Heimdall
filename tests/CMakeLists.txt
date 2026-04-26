# Existing pipeline tests (require GStreamer headers — Jetson/Docker only)
if(TARGET heimdall_pipeline)
    add_executable(heimdall_tests
        test_camera_source.cpp
    )
    target_link_libraries(heimdall_tests
        heimdall_pipeline
        Catch2::Catch2WithMain
    )
    include(Catch)
    catch_discover_tests(heimdall_tests)
endif()

# Tracker tests — no hardware dependency, run on any machine
include(Catch)
add_executable(heimdall_tracker_tests
    test_kalman.cpp
    test_jpda.cpp
    test_tracker.cpp
)
target_link_libraries(heimdall_tracker_tests
    heimdall_tracker
    Catch2::Catch2WithMain
)
catch_discover_tests(heimdall_tracker_tests)

add_executable(heimdall_pose_tests
    test_pose_estimator.cpp
)
target_link_libraries(heimdall_pose_tests
    heimdall_pose
    Catch2::Catch2WithMain
)
catch_discover_tests(heimdall_pose_tests)

add_executable(heimdall_comm_tests
    test_comm_layer.cpp
)
target_link_libraries(heimdall_comm_tests
    heimdall_comm
    libzmq
    cppzmq
    Catch2::Catch2WithMain
)
# On Windows, libzmq is a DLL — add its output dir to the PATH used by catch_discover_tests
# so the post-build test-discovery step can load the executable.
if(WIN32)
    catch_discover_tests(heimdall_comm_tests
        DL_PATHS $<TARGET_FILE_DIR:libzmq>
    )
else()
    catch_discover_tests(heimdall_comm_tests)
endif()
