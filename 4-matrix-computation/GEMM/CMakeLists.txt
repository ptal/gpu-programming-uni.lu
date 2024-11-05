cmake_minimum_required(VERSION 3.26)
project(GEMM LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set Release as the default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CUDA_FLAGS "-Wall -Wextra")
set(compile_options
  $<$<COMPILE_LANGUAGE:CUDA>:
    $<$<CONFIG:Debug>:
      -G --verbose
    >
  >
  $<$<CONFIG:Debug>:
    -g -DDEBUG
  >
  $<$<CONFIG:Release>:
    -O3
  >
)

add_executable(gpu_gemm src/gpu_GEMM.cu)

set_property(TARGET gpu_gemm PROPERTY CUDA_ARCHITECTURES 70)
# If you have several files .cpp, allowing separable compilation is necessary.
set_property(TARGET gpu_gemm APPEND PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(gpu_gemm PRIVATE ${compile_options})
