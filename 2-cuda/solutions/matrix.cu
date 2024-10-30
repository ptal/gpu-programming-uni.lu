// Copyright 2023 Pierre Talbot

#include "utility.hpp"
#include <string>
#include <cassert>

__global__ void gpu_matrix_vector_mul(int n, int m, int** matrix, int* v, int* output) {
  assert(n == gridDim.x * blockDim.x);
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  output[i] = 0;
  for(size_t j = 0; j < m; ++j) {
    output[i] += matrix[i][j] * v[j];
  }
}

__global__ void gpu_matrix_vector_mul_coalesced(int n, int m, int** matrix, int* v, int* output) {
  assert(n == gridDim.x * blockDim.x);
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  output[i] = 0;
  for(size_t j = 0; j < m; ++j) {
    output[i] += matrix[j][i] * v[j];
  }
}

void matrix_vector_mul(const std::vector<std::vector<int>>& matrix, const std::vector<int>& v, std::vector<int>& output) {
  assert(matrix[0].size() == v.size());
  assert(matrix.size() == output.size());
  for(size_t i = 0; i < matrix.size(); ++i) {
    output[i] = 0;
    for(size_t j = 0; j < matrix[i].size(); ++j) {
      output[i] += matrix[i][j] * v[j];
    }
  }
}

int main(int argc, char** argv) {
  size_t m = 1024;
  size_t n = m*m;
  std::vector<std::vector<int>> matrix = initialize_matrix(n, m);
  std::vector<int> v = initialize_vector(m);
  int** gmatrix = initialize_gpu_matrix(matrix);
  int** gmatrix2 = transpose_gpu_matrix(matrix);
  int* gv = initialize_gpu_vector(v);
  int* gv2 = initialize_gpu_vector(v);

  std::vector<int> output(n, 0);
  int* goutput = initialize_gpu_vector(n, 0);
  int* goutput2 = initialize_gpu_vector(n, 0);

  std::cout << "Memory allocated, starting benchmarking." << std::endl;

  // II. Running Matrix-Vector multiplication on CPU.
  long cpu_ms = benchmark_one_ms([&]{
    matrix_vector_mul(matrix, v, output);
  });
  std::cout << "CPU: " << cpu_ms << " ms" << std::endl;

  // III. Running Matrix-Vector multiplication on the GPU grid.
  long gpu_ms_noprefetching = benchmark_one_ms([&]{
    gpu_matrix_vector_mul<<<m, m>>>(n, m, gmatrix, gv, goutput);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU (no prefetching, no coalesced): " << gpu_ms_noprefetching << " ms" << std::endl;
  long gpu_ms_prefetched = benchmark_one_ms([&]{
    gpu_matrix_vector_mul<<<m, m>>>(n, m, gmatrix, gv, goutput);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU (prefetched, no coalesced): " << gpu_ms_prefetched << " ms" << std::endl;

  long gpu_ms_noprefetching_coalesced = benchmark_one_ms([&]{
    gpu_matrix_vector_mul_coalesced<<<m, m>>>(n, m, gmatrix2, gv2, goutput2);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU (no prefetching, coalesced): " << gpu_ms_noprefetching_coalesced << " ms" << std::endl;
  long gpu_ms_prefetched_coalesced = benchmark_one_ms([&]{
    gpu_matrix_vector_mul_coalesced<<<m, m>>>(n, m, gmatrix2, gv2, goutput2);
    CUDIE(cudaDeviceSynchronize());
  });
  std::cout << "GPU (prefetched, coalesced): " << gpu_ms_prefetched_coalesced << " ms" << std::endl;

  // IV. Verifying both give the same result (TODO: add deallocation).
  check_equal_vector(output, goutput2);
  return 0;
}
