#include <cstdlib>
#include <iostream>

#include "utility.hpp"
#include "datatypes.hpp"

template<typename T>
using gemm_fcn = void (*)(
  T const* const A, T const* const B, T* const C,
  size_t const M, size_t const K, size_t const N,
  size_t const ldA, size_t const ldB, size_t const ldC
);

template<typename T>
void gpu_GEMM(
  DenseMatrix<T> const& mtxA, DenseMatrix<T> const& mtxB, DenseMatrix<T>& mtxC,
  size_t const m_block_size, size_t const n_block_size
);

template<typename T>
void gpu_GEMM(
  DenseMatrix<T> const& mtxA, DenseMatrix<T> const& mtxB, DenseMatrix<T>& mtxC,
  size_t const m_block_size, size_t const n_block_size,
  gemm_fcn<T> gemm
) {
  T* A;
  T* B;
  T* C;
  // Allocate pinned memory
  try_CUDA(cudaMallocHost(&A, sizeof(T) * mtxA.nnz));
  try_CUDA(cudaMallocHost(&B, sizeof(T) * mtxB.nnz));
  try_CUDA(cudaMallocHost(&C, sizeof(T) * mtxC.nnz));

  size_t m_grid_size = mtxC.m/m_block_size;
  if ( mtxC.m % m_block_size != 0 ) m_grid_size++;
  size_t n_grid_size = mtxC.n/n_block_size;
  if ( mtxC.n % n_block_size != 0 ) n_grid_size++;

  dim3 threadsPerBlock(m_block_size, n_block_size);
  dim3 blocksPerGrid(m_grid_size, n_grid_size);

  try_CUDA( cudaMemcpy(A, mtxA.data, sizeof(T) * mtxA.nnz, cudaMemcpyHostToDevice) );
  try_CUDA( cudaMemcpy(B, mtxB.data, sizeof(T) * mtxB.nnz, cudaMemcpyHostToDevice) );

  gemm<<<threadsPerBlock,blocksPerGrid>>>(
    A, B, C,
    mtxA.m, mtxA.n, mtxB.m,
    mtxA.ld, mtxB.ld, mtxC.ld
  );
  cudaDeviceSynchronize();

  try_CUDA( cudaMemcpy(mtxC.data, C, sizeof(T) * mtxC.nnz, cudaMemcpyDeviceToHost) );

  try_CUDA( cudaFreeHost(A) );
  try_CUDA( cudaFreeHost(B) );
  try_CUDA( cudaFreeHost(C) );
}

template<typename T>
__global__ void gpu_GEMM_naive(
  T const* const A, T const* const B, T* const C,
  size_t const M, size_t const K, size_t const N,
  size_t const ldA, size_t const ldB, size_t const ldC
);

template<typename T>
__global__ void gpu_GEMM_naive(
  T const* const A, T const* const B, T* const C,
  size_t const M, size_t const K, size_t const N,
  size_t const ldA, size_t const ldB, size_t const ldC
) {
  size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t const j = blockIdx.y * blockDim.y + threadIdx.y;

  if ( i > M || j > N ) return;

  T c_ij = static_cast<T>(0);

  for ( size_t k = 0; k < K; k++ ) {
    T const a_ik = A[i + ldA*k];
    T const b_kj = B[k + ldB*j];
    c_ij += a_ik * b_kj;
  }

  C[i + ldC*j] = c_ij;
}

int main() {
  DenseMatrix<float> mtxA(4*6);
  for (size_t i = 0; i < mtxA.nnz; i++) {
    mtxA.data[i] = i;
  }
  mtxA.m = 4;
  mtxA.n = 4;
  mtxA.ld = 6;

  DenseMatrix<float> mtxB(4*5);
  for (size_t i = 0; i < mtxB.nnz; i++) {
    mtxB.data[i] = 0;
  }
  mtxB.m = 4;
  mtxB.n = 5;
  mtxB.ld = 5;
  mtxB.data[0 + 0*mtxB.ld] = 1.0;
  mtxB.data[1 + 1*mtxB.ld] = 1.0;
  mtxB.data[2 + 2*mtxB.ld] = 1.0;
  mtxB.data[3 + 3*mtxB.ld] = 1.0;

  DenseMatrix<float> mtxC(4*5);
  mtxC.m = 4;
  mtxC.n = 5;
  mtxC.ld = 5;

  gpu_GEMM(mtxA, mtxB, mtxC, 2, 2, gpu_GEMM_naive);

  std::cout << "Matrix A:" << "\n";
  mtxA.display(std::cout);
  std::cout << "Matrix B:" << "\n";
  mtxB.display(std::cout);
  std::cout << "Matrix C:" << "\n";
  mtxC.display(std::cout);

  return EXIT_SUCCESS;
}
