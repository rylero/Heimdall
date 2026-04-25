find_package(CUDA REQUIRED)

add_library(rfdetr_parser SHARED)
target_include_directories(rfdetr_parser PRIVATE
    "${DS_ROOT}/sources/includes"
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(rfdetr_parser ${CUDA_LIBRARIES})
