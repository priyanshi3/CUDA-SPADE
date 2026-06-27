# pybind11 and Python3 are discovered in the root CMakeLists.txt before this
# subdirectory is added, so pybind11_add_module is already available here.

# pybind11_add_module does three things automatically:
#   1. Compiles spade_bindings.cpp as a shared library
#   2. Sets the correct filename suffix (.pyd on Windows, .so on Linux)
#   3. Links against the right Python version's import library
pybind11_add_module(spade_cuda spade_bindings.cpp)

# spade_bindings.cpp calls functions declared in spade.h
target_include_directories(spade_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Link against the CUDA library built in src/ — this pulls in cudart and cufft
# transitively (because src/CMakeLists.txt already links them into spade)
target_link_libraries(spade_cuda PRIVATE spade)
