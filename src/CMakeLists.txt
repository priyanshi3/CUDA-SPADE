find_package(CUDAToolkit REQUIRED)

add_library(spade STATIC
    spade.cu
)

target_include_directories(spade PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(spade
    CUDA::cudart          # CUDA runtime
    CUDA::cufft           # FFT library (used in Phase 3)
)
