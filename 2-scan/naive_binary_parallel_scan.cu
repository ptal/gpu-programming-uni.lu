#include <cstdlib>
#include <iostream>
#include <random>
#include <limits>
#include <vector>
#include <algorithm>
#include <cstdint>

#include "utility.hpp"

template<typename T>
void append_random( T& v, size_t n, std::seed_seq& s ) {
  std::mt19937 mersenne_generator{s};
  std::uniform_int_distribution<int> distribution{0, std::numeric_limits<int>::max()};
  std::generate_n(
    std::back_inserter(v),
    n,
    [&distribution, &mersenne_generator]() {
      // See: https://en.wikipedia.org/wiki/Inverse_transform_sampling
      return distribution(mersenne_generator);
    }
  );
}

template<typename T>
inline void append_random( T& v, size_t n, std::seed_seq&& seed ) {
  append_random<T>(v, n, seed);
}

template<typename T>
void compare_vectors( T const* const v0, T const* const v1, size_t const n )
{
  for ( size_t i =0; i < n; i++ ) {
    if ( v0[i] != v1[i] ) {
      std::cout << "Error in position " << std::to_string(i)
        << ": " << std::to_string(v0[i]) << " != " << std::to_string(v1[i])
        << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

template<typename T>
void scan( std::vector<T>& v ) {
  for ( size_t i = 1; i < v.size(); i++ ) {
    v[i] += v[i-1];
  }
}

template<typename T>
__device__ void update_sums( T* const v, size_t const n, size_t const p, size_t const p_next ) {
  size_t k = threadIdx.x;
  if ( k >= p && k < n ) {
    v[k] += v[k - p];
  }
}

template<typename T>
__global__ void naive_parallel_binary_scan( T* const v, size_t const n ) {
  size_t p = 1;
  size_t p_next = 2*p;
  while ( p < n ) {
    update_sums( v, n, p, p_next );
    p = p_next;
    p_next = 2*p;
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  if ( argc != 2 ) {
    std::cout << "Usage: " << argv[0] << " <vector size>" << std::endl;
    exit(EXIT_FAILURE);
  }

  size_t n = std::stoll(argv[1]);

  // 1. Initialize a vector
  std::vector<int64_t> v_cpu;
  int64_t seed_value = 0;
  //int64_t seed_value = static_cast<int64_t>(std::random_device{}());
  append_random(v_cpu, n, std::seed_seq{seed_value});

  // 2. Initialize GPU global vector and set up data transfer to and from the GPU
  int64_t* v_gpu_synced;
  try_cuda( cudaMallocManaged( &v_gpu_synced, n * sizeof( int64_t ) ) );

  // 3. Copy data to the GPU
  std::copy( v_cpu.begin(), v_cpu.end(), v_gpu_synced );

  naive_parallel_binary_scan<<<1, n>>>( v_gpu_synced, n );
  try_cuda( cudaDeviceSynchronize() );

  // 4. Test results
  scan( v_cpu );
  compare_vectors( v_cpu.data(), v_gpu_synced, n );

  // 5. Free the GPU global vector
  try_cuda( cudaFree(v_gpu_synced) );

  return 0;
}
