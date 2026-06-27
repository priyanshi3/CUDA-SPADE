cmake_minimum_required(VERSION 3.20)
project(cuda_spade CUDA CXX)

# ── CUDA settings ─────────────────────────────────────────────────────────────
# sm_86 = RTX 30xx / RTX 40xx Ada Lovelace (your RTX 4060)
# sm_89 = RTX 40xx Ada Lovelace (if yours is 40-series, both are fine to include)
set(CMAKE_CUDA_ARCHITECTURES 86 89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Useful CUDA compile flags:
#   --expt-relaxed-constexpr  — lets constexpr functions run on GPU
#   --use_fast_math           — faster but slightly less precise math (fine for us)
#   -lineinfo                 — keeps line numbers for Nsight profiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG   "-g -G")       # Full debug info (slow)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")         # Max optimisation

add_subdirectory(src)
add_subdirectory(tests)

# ── Python bindings (optional) ────────────────────────────────────────────────
# Builds a .pyd (Windows) / .so (Linux) that Python can import as spade_cuda.
# Requires pybind11:  pip install pybind11
#
# Discovery: ask the active Python interpreter where pybind11 put its CMake
# config, then use that path with find_package.
find_package(Python3 QUIET COMPONENTS Interpreter Development)
if(Python3_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pb11_ret
        ERROR_QUIET
    )
    if(_pb11_ret EQUAL 0 AND EXISTS "${pybind11_DIR}")
        find_package(pybind11 CONFIG REQUIRED PATHS "${pybind11_DIR}")
        add_subdirectory(python)
        message(STATUS "Python bindings: ENABLED  (spade_cuda module will be built)")
    else()
        message(STATUS "Python bindings: DISABLED (pybind11 not found — run: pip install pybind11)")
    endif()
else()
    message(STATUS "Python bindings: DISABLED (Python3 not found)")
endif()
