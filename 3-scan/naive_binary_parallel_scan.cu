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
void test_unary_sum( T const* const v, size_t const n )
{
  for ( size_t i =0; i < n; i++ ) {
    if ( v[i] != static_cast<T>(i + 1) )  {
      std::cout << "Error in position " << std::to_string(i)
        << ": " << std::to_string(v[i]) << " != " << std::to_string(i+1)
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
__forceinline__ __device__ void update_sums( T* const v, size_t const n, size_t const p, size_t const p_next ) {
  size_t const k = threadIdx.x;
  size_t const offset = blockIdx.x * blockDim.x;
  size_t const idx = offset + k;
  if ( k >= p && idx < n ) {
    v[idx] += v[idx - p];
  }
}

template<typename T>
__device__ void block_naive_parallel_binary_scan( T* const v, size_t const n ) {
  //size_t const idx = threadIdx.x + blockIdx.x * blockDim.x;

  //__syncthreads();
  //if ( idx == 0 ) {
  //  printf("\n\nPRE\n\n");
  //  for (uint64_t i = 0; i < blockDim.x*gridDim.x && i < n; i++ ) {
  //    printf("%llu: %lld \n", i, v[i]);
  //  }
  //}
  //__syncthreads();

  size_t p = 1;
  size_t p_next = 2*p;
  while ( p < n && p < blockDim.x ) {
    update_sums( v, n, p, p_next );
    p = p_next;
    p_next = 2*p;
    __syncthreads();
  }

  //__syncthreads();
  //if ( idx == 0 ) {
  //  printf("\n\nPOST\n\n");
  //  for (uint64_t i = 0; i < blockDim.x*gridDim.x && i < n; i++ ) {
  //    printf("%llu: %lld \n", i, v[i]);
  //  }
  //}
  //__syncthreads();
}

template<typename T>
__global__ void step_block_naive_parallel_binary_scan( T* const v, size_t const n, T* const v_cumulative ) {
  block_naive_parallel_binary_scan( v, n );

  size_t const idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( threadIdx.x + 1 == blockDim.x && idx < n ) {
    v_cumulative[blockIdx.x] = v[idx];
    //printf("%llu = %llu * %llu + %llu mod %llu: %lld \n", idx, static_cast<uint64_t>(blockIdx.x), static_cast<uint64_t>(blockDim.x), static_cast<uint64_t>(threadIdx.x), static_cast<uint64_t>(blockDim.x), v[idx]);
  }

  //__syncthreads();
  //  printf("\n\nHERE\n\n");
  //if ( idx == 0 ) {
  //  for (uint64_t i = 0; i < blockDim.x*gridDim.x && i < n; i++ ) {
  //    printf("%llu, %llu: %lld, %lld \n", i, static_cast<uint64_t>(i/blockDim.x), v[i], v_cumulative[i/blockDim.x]);
  //  }
  //}
  //__syncthreads();
}

template<typename T>
__global__ void single_block_naive_parallel_binary_scan( T* const v, size_t const n ) {
  block_naive_parallel_binary_scan( v, n );
}

template<typename T>
__global__ void add_subblocks( T* const v, size_t const n, T const* const increments ) {
  size_t const idx = threadIdx.x + blockIdx.x * blockDim.x; 

  //printf("%llu, %llu: %lld \n", idx, static_cast<uint64_t>(blockIdx.x), increments[blockIdx.x]);

  //__syncthreads();
  //if ( idx == 0 ) {
  //  printf("\n\nPRE\n\n");
  //  for (uint64_t i = 0; i < blockDim.x*gridDim.x && i < n; i++ ) {
  //    printf("%llu, %llu: %lld, %lld \n", i, static_cast<uint64_t>(i/blockDim.x), v[i], increments[i/blockDim.x]);
  //  }
  //}
  //__syncthreads();
  if ( blockIdx.x > 0 && idx < n ) {
    v[idx] += increments[blockIdx.x-1];
  }
  //__syncthreads();
  //if ( idx == 0 ) {
  //  printf("\n\nPOST\n\n");
  //  for (uint64_t i = 0; i < blockDim.x*gridDim.x && i < n; i++ ) {
  //    printf("%llu, %llu: %lld, %lld \n", i, static_cast<uint64_t>(i/blockDim.x), v[i], increments[i/blockDim.x]);
  //  }
  //}
  //__syncthreads();
}

template<typename T>
void naive_parallel_binary_scan( T* const v, size_t const n, size_t const block_size )
{
  size_t n_blocks = n / block_size;
  size_t const v_comulative_size = n_blocks;
  if ( n % block_size !=0 ) {
    n_blocks++;
  }
  dim3 const threadsPerBlock(block_size);

  if ( n_blocks <= 1 ) {
    dim3 const numBlocks(1);
    single_block_naive_parallel_binary_scan<<<numBlocks,threadsPerBlock>>>(v, n);
  } else {
    T* v_comulative;
    try_cuda( cudaMalloc(&v_comulative, v_comulative_size * sizeof(T) ) );

    dim3 const numBlocks(n_blocks);
    step_block_naive_parallel_binary_scan<<<n_blocks, block_size>>>(v, n, v_comulative);

    naive_parallel_binary_scan( v_comulative, n_blocks, block_size );

    add_subblocks<<<n_blocks, block_size>>>( v, n, v_comulative );

    try_cuda( cudaFree(v_comulative) );
  }
}

int main( int argc, char** argv ) {
  if ( argc != 3 ) {
    std::cout << "Usage: " << argv[0]
      << " <vector size>"
      << " <block size>"
      << std::endl;
    exit(EXIT_FAILURE);
  }

  size_t n = std::stoll(argv[1]);
  size_t block_size = std::stoll(argv[2]);

  // 1. Initialize a vector
  std::vector<int64_t> v_cpu;
  //// Random initialization
  //int64_t seed_value = static_cast<int64_t>(std::random_device{}());
  //// Random initialization with fixed seed
  //int64_t seed_value = 0;
  //append_random(v_cpu, n, std::seed_seq{seed_value});
  // Initialization with ones for debugging
  std::fill_n(std::back_inserter(v_cpu), n, 1);

  // 2. Initialize GPU global vector and set up data transfer to and from the GPU
  int64_t* v_gpu_synced;
  try_cuda( cudaMallocManaged( &v_gpu_synced, n * sizeof( int64_t ) ) );

  // 3. Copy data to the GPU
  std::copy( v_cpu.begin(), v_cpu.end(), v_gpu_synced );

  naive_parallel_binary_scan( v_gpu_synced, n, block_size );
  try_cuda( cudaDeviceSynchronize() );

  // 4. Test results
  //scan( v_cpu );
  //compare_vectors( v_cpu.data(), v_gpu_synced, n );
  test_unary_sum(v_gpu_synced, n);

  // 5. Free the GPU global vector
  try_cuda( cudaFree(v_gpu_synced) );

  return 0;
}
