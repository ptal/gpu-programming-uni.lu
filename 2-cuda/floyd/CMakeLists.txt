cmake_minimum_required(VERSION 3.20)

include(FetchContent)
include(CMakePrintHelpers)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
# Fix found at https://github.com/ethereum/solidity/pull/13429
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

project(CudaMaster LANGUAGES CUDA CXX)

# Using C++17, you can also use C++20 if nvcc --version >= 12.

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# An example of dependency.

FetchContent_Declare(
  cuda_battery
  GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
  GIT_TAG        v1.5.3
)
FetchContent_MakeAvailable(cuda_battery)

# We are allowed to write C++ files with .cpp extension instead of .cu.
set_source_files_properties(
    src/floyd.cpp src/config.cpp
    PROPERTIES LANGUAGE CUDA)

# CUDA Master executable

add_executable(floyd src/floyd.cpp src/config.cpp)
target_include_directories(floyd PRIVATE include ${cuda_battery_SOURCE_DIR}/include)

set_property(TARGET floyd PROPERTY CUDA_ARCHITECTURES 70)

# If you have several files .cpp, allowing separable compilation is necessary.
set_property(TARGET floyd APPEND PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(floyd PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        $<$<CONFIG:Debug>:-G --verbose>>
    $<$<CONFIG:Debug>:-g -DDEBUG>
)
