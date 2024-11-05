// Copyright 2023 Pierre Talbot

#include <cstdlib>
#include <cstdio>
#include <chrono>

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
