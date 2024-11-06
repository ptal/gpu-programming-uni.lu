// Copyright 2023 Pierre Talbot

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <random>
#include <limits>
#include <iterator>
#include <algorithm>

#pragma once

inline static void handle_CUDA_error( cudaError_t err,
                                 const char *file,
                                 int line) {
  if ( err != cudaSuccess ) {
    printf( "%s:%d CUDA runtime error %s\n", file, line, cudaGetErrorString( err ) );
    exit( EXIT_FAILURE );
  }
}

#define try_CUDA( result ) handle_CUDA_error( (result), __FILE__, __LINE__ )

/** Benchmarks the time taken by the function `f` by executing it 1 time first
 * for warm-up, then 10 times in a row, then dividing the result by 10.
 * It returns the duration in milliseconds. */
template<class F>
double benchmark_averaged_ms(F&& f, unsigned long long int n) {
  f(); // warm-up.

  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < n; ++i) {
    f();
  }
  auto end = std::chrono::steady_clock::now();

  std::chrono::milliseconds Dt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return static_cast<double>( Dt.count() /* This will be a unint64_t */ ) / static_cast<double>(n);
}

template<typename T>
void append_random( T& v, size_t const n, std::seed_seq& s ) {
  std::mt19937 mersenne_generator{s};
  std::uniform_int_distribution<int> distribution{0, std::numeric_limits<int>::max()};
  std::generate_n(
    std::back_inserter(v),
    n,
    [&distribution, &mersenne_generator]() {
    // See: https://en.wikipedia.org/wiki/Inverse_transform_sampling
      return static_cast<T>(distribution(mersenne_generator));
    }
  );
}

template<typename T>
inline void append_random( T& v, size_t const n, std::seed_seq&& s ) {
  append_random<T>(v, n, s);
}

template<typename T>
void fill_random( T*& v, size_t const n, std::seed_seq& s ) {
  std::mt19937 mersenne_generator{s};
  std::uniform_int_distribution<int> distribution{0, 100};
  std::generate_n(
    v,
    n,
    [&distribution, &mersenne_generator]() {
      return static_cast<T>(distribution(mersenne_generator));
    }
  );
}

template<typename T>
inline void fill_random( T*& v, size_t const n, std::seed_seq&& s ) {
  fill_random<T>(v, n, s);
}

