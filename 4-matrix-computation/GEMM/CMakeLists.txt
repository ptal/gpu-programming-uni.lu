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

include(CMakePrintHelpers)

find_package(CUDA REQUIRED)
find_package(BLAS REQUIRED)
set(BLAS_ROOT_DIRS "")
foreach(lib ${BLAS_LIBRARIES})
  cmake_path(GET lib PARENT_PATH lib_dir)
  cmake_path(GET lib_dir PARENT_PATH base_dir)
  list(APPEND BLAS_ROOT_DIRS "${base_dir}/include")
  list(APPEND BLAS_ROOT_DIRS "${base_dir}/include/ilp64")
  list(APPEND BLAS_ROOT_DIRS "${base_dir}/include/lp64")
endforeach()
find_path(BLAS_INCLUDE_DIRS cblas.h ${BLAS_ROOT_DIRS})

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
target_link_libraries(gpu_gemm PRIVATE BLAS::BLAS ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
target_include_directories(gpu_gemm PRIVATE ${BLAS_INCLUDE_DIRS})
