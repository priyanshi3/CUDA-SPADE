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
