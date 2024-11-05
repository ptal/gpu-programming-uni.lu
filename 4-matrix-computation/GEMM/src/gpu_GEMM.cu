#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <random>
#include <cmath>

#include <cblas.h>

#include "utility.hpp"
#include "datatypes.hpp"

template<typename T>
using gemm_fcn = void (*)(
  T const* const A, T const* const B, T* const C,
  size_t const M, size_t const N, size_t const K,
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
    mtxA.m, mtxA.n, mtxB.n,
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
  size_t const M, size_t const N, size_t const K,
  size_t const ldA, size_t const ldB, size_t const ldC
);

template<typename T>
__global__ void gpu_GEMM_naive(
  T const* const A, T const* const B, T* const C,
  size_t const M, size_t const N, size_t const K,
  size_t const ldA, size_t const ldB, size_t const ldC
) {
  size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t const j = blockIdx.y * blockDim.y + threadIdx.y;

  if ( i > M || j > K ) return;

  T c_ij = static_cast<T>(0);

  for ( size_t l = 0; l < N; l++ ) {
    T const a_ik = A[i + ldA*l];
    T const b_kj = B[l + ldB*j];
    c_ij += a_ik * b_kj;
  }

  C[i + ldC*j] = c_ij;
}

float f_norm(DenseMatrix<float> const& mtx) {
  float norm = 0.0;
  for ( size_t i = 0; i < mtx.n; i++ ) {
    // float cblas_snrm2(const CBLAS_INT N, const float *X, const CBLAS_INT incX);
    float const delta = cblas_snrm2( mtx.m, &mtx.data[i*mtx.ld], 1);
    norm += delta*delta;
  }

  return std::sqrt(norm);
}

void matrix_saxpy(
  size_t const N,
  float const alpha,
  float const* A, size_t const lda, float* B, size_t const ldb
) {
  size_t const incX = 1;
  size_t const incY = 1;

  for (size_t i = 0; i < N; i++) {
    float const* const X = &A[i*lda];
    float*  const Y = &B[i*ldb];

    // cblas_saxpy(
    //   const CBLAS_INT N, const float alpha, const float *X,
    //   const CBLAS_INT incX, float *Y, const CBLAS_INT incY
    // );
    cblas_saxpy(N, alpha, X, incX, Y, incY);
  }
}

float gemm_error(DenseMatrix<float> const& mtxA, DenseMatrix<float> const& mtxB, DenseMatrix<float> const& mtxC) {
  DenseMatrix<float> accu(mtxC.nnz);
  std::fill_n(accu.data, accu.nnz, 0.0);
  accu.ld = mtxC.ld;
  accu.m = mtxC.m;
  accu.n = mtxC.n;

  size_t const M = mtxA.m;
  size_t const N = mtxA.n;
  size_t const K = mtxB.n;

  float alpha = 1.0;
  float beta = 0.0;

  // void cblas_sgemm(
  //   CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
  //   const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT K,
  //   const float alpha,
  //   const float *A, const CBLAS_INT lda, const float *B, const CBLAS_INT ldb,
  //   const float beta, float *C, const CBLAS_INT ldc
  // );
  cblas_sgemm(
    CblasColMajor, CblasNoTrans, CblasNoTrans,
    M, N, K,
    alpha,
    mtxA.data, mtxA.ld, mtxB.data, mtxB.ld,
    beta, accu.data, accu.ld
  );

  // accu <- accu - C
  matrix_saxpy(
    K,
    -1.0,
    mtxC.data, mtxC.ld, accu.data, accu.ld
  );

  return f_norm(accu);
}

int main() {
  DenseMatrix<float> mtxA(4*6);
  for (size_t i = 0; i < mtxA.nnz; i++) {
    mtxA.data[i] = i;
  }
  mtxA.m = 4;
  mtxA.n = 5;
  mtxA.ld = 6;

  DenseMatrix<float> mtxB(5*5);
  for (size_t i = 0; i < mtxB.nnz; i++) {
    mtxB.data[i] = 0;
  }
  mtxB.m = 5;
  mtxB.n = 4;
  mtxB.ld = 5;
  mtxB.data[0 + 0*mtxB.ld] = 1.0;
  mtxB.data[1 + 1*mtxB.ld] = 1.0;
  mtxB.data[2 + 2*mtxB.ld] = 1.0;
  mtxB.data[3 + 3*mtxB.ld] = 1.0;

  DenseMatrix<float> mtxC(5*5);
  mtxC.m = 5;
  mtxC.n = 4;
  mtxC.ld = 5;
  std::fill_n(mtxC.data, mtxC.nnz, 0.0);

  gpu_GEMM(mtxA, mtxB, mtxC, 2, 2, gpu_GEMM_naive);

  std::cout << "Matrix A:" << "\n";
  mtxA.display(std::cout);
  std::cout << "Matrix B:" << "\n";
  mtxB.display(std::cout);
  std::cout << "Matrix C:" << "\n";
  mtxC.display(std::cout);

  std::cout << "Error (Frobenius norm): "
            << gemm_error(mtxA, mtxB, mtxC)
            << "\n";

  return EXIT_SUCCESS;
}
