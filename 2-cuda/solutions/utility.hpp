// Copyright 2023 Pierre Talbot

#include <cstdio>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>

#ifndef UTILITY_HPP
#define UTILITY_HPP

#define CUDIE(result) { \
  cudaError_t e = (result); \
  if (e != cudaSuccess) { \
    printf("%s:%d CUDA runtime error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }}

/** Initialize a matrix of size `n` with elements between 0 and 99. */
template <class T>
std::vector<std::vector<T>> initialize_matrix_gen(size_t n, size_t m) {
  std::vector<std::vector<T>> matrix(n, std::vector<T>(m));
  // std::mt19937 m{std::random_device{}()};
  std::mt19937 r{0}; // fixed seed to ease debugging.
  std::uniform_int_distribution<T> dist{0, 99};
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < m; ++j) {
      matrix[i][j] = dist(r);
    }
  }
  return std::move(matrix);
}

std::vector<std::vector<int>> initialize_matrix(size_t n, size_t m) {
  return initialize_matrix_gen<int>(n, m);
}

/** Initialize an array of size `n` in managed memory. */
int* init_random_vector(size_t n) {
  int* v;
  CUDIE(cudaMallocManaged(&v, sizeof(int) * n));
  std::mt19937 m{std::random_device{}()};
  std::uniform_int_distribution<int> dist{1, std::numeric_limits<int>::max()};
  std::generate(v, v + n, [&dist, &m](){return dist(m);});
  return v;
}

std::vector<int> initialize_vector(size_t n) {
  std::vector<int> v(n);
  std::mt19937 m{std::random_device{}()};
  std::uniform_int_distribution<int> dist{1, std::numeric_limits<int>::max()};
  std::generate(v.begin(), v.end(), [&dist, &m](){return dist(m);});
  return std::move(v);
}

/** Compare two matrices to ensure they are equal. */
template <class T>
void check_equal_matrix(const std::vector<std::vector<T>>& matrix, T** gpu_matrix) {
  for(size_t i = 0; i < matrix.size(); ++i) {
    for(size_t j = 0; j < matrix.size(); ++j) {
      if(matrix[i][j] != gpu_matrix[i][j]) {
        printf("Found an error: %d != %d\n", matrix[i][j], gpu_matrix[i][j]);
        exit(1);
      }
    }
  }
}

/** Compare two matrices to ensure they are equal. */
template <class T>
void check_equal_vector(const std::vector<T>& cpu_vector, T* gpu_vector) {
  for(size_t i = 0; i < cpu_vector.size(); ++i) {
    if(cpu_vector[i] != gpu_vector[i]) {
      printf("Found an error: %d != %d\n", cpu_vector[i], gpu_vector[i]);
      exit(1);
    }
  }
}

/** Copy a CPU matrix to the managed memory of the GPU. */
template <class T>
T** initialize_gpu_matrix(const std::vector<std::vector<T>>& m) {
  size_t n = m.size();
  T** gpu_matrix;
  CUDIE(cudaMallocManaged(&gpu_matrix, sizeof(T*) * n));
  for(int i = 0; i < n; ++i) {
    CUDIE(cudaMallocManaged(&gpu_matrix[i], sizeof(T) * m[i].size()));
  }
  for(int i = 0; i < m.size(); ++i) {
    for(int j = 0; j < m[i].size(); ++j) {
      gpu_matrix[i][j] = m[i][j];
    }
  }
  return gpu_matrix;
}

template <class T>
T** transpose_gpu_matrix(const std::vector<std::vector<T>>& m) {
  T** gpu_matrix;
  CUDIE(cudaMallocManaged(&gpu_matrix, sizeof(T*) * m[0].size()));
  for(int i = 0; i < m[0].size(); ++i) {
    CUDIE(cudaMallocManaged(&gpu_matrix[i], sizeof(T) * m.size()));
  }
  for(int i = 0; i < m.size(); ++i) {
    for(int j = 0; j < m[i].size(); ++j) {
      gpu_matrix[j][i] = m[i][j];
    }
  }
  return gpu_matrix;
}

template <class T>
T* initialize_gpu_vector(const std::vector<T>& v) {
  size_t n = v.size();
  T* gv;
  CUDIE(cudaMallocManaged(&gv, sizeof(T) * n));
  for(int i = 0; i < v.size(); ++i) {
    gv[i] = v[i];
  }
  return gv;
}

template <class T>
T* initialize_gpu_vector(size_t n, const T& def) {
  T* gv;
  CUDIE(cudaMallocManaged(&gv, sizeof(T) * n));
  for(int i = 0; i < n; ++i) {
    gv[i] = def;
  }
  return gv;
}

/** Deallocating the GPU matrix. */
template <class T>
void deallocate_gpu_distances(T** gpu_matrix, size_t n) {
  for(int i = 0; i < n; ++i) {
    cudaFree(gpu_matrix[i]);
  }
  cudaFree(gpu_matrix);
}

/** Benchmarks the time taken by the function `f` by executing it 1 time first for warm-up, then 10 times in a row, then dividing the result by 10.
 * It returns the duration in milliseconds. */
template<class F>
long benchmark_ms(F&& f) {
  f(); // warm-up.
  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < 10; ++i) {
    f();
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 10;
}

template<class F>
long benchmark_one_ms(F&& f) {
  auto start = std::chrono::steady_clock::now();
  f();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

#endif
