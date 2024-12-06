#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <random>
#include <cmath>

#include <cblas.h>
#include <cublasXt.h>

#include "utility.hpp"
#include "datatypes.hpp"

#define BLOCK_SIDE 16
#define M_BLOCK_SIZE BLOCK_SIDE
#define N_BLOCK_SIZE BLOCK_SIDE

template<typename T>
using gemm_fcn = void (*)(
  size_t const M, size_t const N, size_t const K,
  T const alpha,
  T const* const A, size_t const ldA, T const* const B, size_t const ldB,
  T const beta,
  T* const C, size_t const ldC
);

__host__ __device__ inline size_t quotient_ceiling( size_t const dimension, size_t const block_size ) {
  size_t q = dimension / block_size;

  if ( dimension % block_size != 0 ) {
    q++;
  }

  return q;
}

template<typename T>
void gpu_GEMM(
  T const alpha,
  DenseMatrix<T> const& mtxA, DenseMatrix<T> const& mtxB,
  T const beta,
  DenseMatrix<T>& mtxC,
  gemm_fcn<T> gemm,
  float& duration
) {
  T* A;
  T* B;
  T* C;
  // Allocate pinned memory
  try_CUDA(cudaMallocHost(&A, sizeof(T) * mtxA.nnz));
  try_CUDA(cudaMallocHost(&B, sizeof(T) * mtxB.nnz));
  try_CUDA(cudaMallocHost(&C, sizeof(T) * mtxC.nnz));

  size_t const M = mtxC.m;
  size_t const N = mtxC.n;
  size_t const K = mtxA.n;

  size_t const m_grid_size = quotient_ceiling( mtxC.m, M_BLOCK_SIZE);
  size_t const n_grid_size = quotient_ceiling( mtxC.n, N_BLOCK_SIZE);

  dim3 const threadsPerBlock(M_BLOCK_SIZE, N_BLOCK_SIZE);
  dim3 const blocksPerGrid(m_grid_size, n_grid_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  try_CUDA( cudaMemcpy(A, mtxA.data, sizeof(T) * mtxA.nnz, cudaMemcpyHostToDevice) );
  try_CUDA( cudaMemcpy(B, mtxB.data, sizeof(T) * mtxB.nnz, cudaMemcpyHostToDevice) );

  gemm<<<blocksPerGrid, threadsPerBlock>>>(
    M, N, K,
    alpha,
    A, mtxA.ld, B, mtxB.ld,
    beta,
    C, mtxC.ld
  );

  try_CUDA( cudaMemcpy(mtxC.data, C, sizeof(T) * mtxC.nnz, cudaMemcpyDeviceToHost) );

  cudaEventRecord(stop);

  try_CUDA( cudaFreeHost(A) );
  try_CUDA( cudaFreeHost(B) );
  try_CUDA( cudaFreeHost(C) );

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);
}

template<typename T>
__global__ void gpu_GEMM_naive(
  size_t const M, size_t const N, size_t const K,
  T const alpha,
  T const* const A, size_t const ldA, T const* const B, size_t const ldB,
  T const beta,
  T* const C, size_t const ldC
) {
  size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t const j = blockIdx.y * blockDim.y + threadIdx.y;

  if ( i >= M || j >= N ) return;

  T c_ij = static_cast<T>(0);

  for ( size_t l = 0; l < K; l++ ) {
    T const a_il = A[i + ldA*l];
    T const b_lj = B[l + ldB*j];
    c_ij += a_il * b_lj;
  }

  size_t const idx_C_ij = i + ldC*j;
  C[idx_C_ij] = alpha * c_ij + beta * C[idx_C_ij];
}

template<typename T>
__global__ void gpu_GEMM_coalesce(
  size_t const M, size_t const N, size_t const K,
  T const alpha,
  T const* const A, size_t const ldA, T const* const B, size_t const ldB,
  T const beta,
  T* const C, size_t const ldC
) {
  size_t const i = blockIdx.x * blockDim.x + threadIdx.y;
  size_t const j = blockIdx.y * blockDim.y + threadIdx.x;

  if ( i >= M || j >= N ) return;

  T c_ij = static_cast<T>(0);

  for ( size_t l = 0; l < K; l++ ) {
    T const a_il = A[i + ldA*l];
    T const b_lj = B[l + ldB*j];
    c_ij += a_il * b_lj;
  }

  size_t const idx_C_ij = i + ldC*j;
  C[idx_C_ij] = alpha * c_ij + beta * C[idx_C_ij];
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
  size_t const M, size_t const N,
  float const alpha,
  float const* A, size_t const lda, float* B, size_t const ldb
) {
  size_t const incX = 1;
  size_t const incY = 1;

  for (size_t i = 0; i < N; i++) {
    float const* const X = &A[i*lda];
    float* const Y = &B[i*ldb];

    // cblas_saxpy(
    //   const CBLAS_INT N, const float alpha, const float *X,
    //   const CBLAS_INT incX, float *Y, const CBLAS_INT incY
    // );
    cblas_saxpy(M, alpha, X, incX, Y, incY);
  }
}

float gemm_error(
  float const alpha,
  DenseMatrix<float> const& mtxA, DenseMatrix<float> const& mtxB,
  float const beta,
  DenseMatrix<float> const& mtxC
) {
  DenseMatrix<float> accu(mtxC.nnz);
  std::fill_n(accu.data, accu.nnz, 0.0);
  accu.ld = mtxC.ld;
  accu.m = mtxC.m;
  accu.n = mtxC.n;

  size_t const M = mtxC.m;
  size_t const N = mtxC.n;
  size_t const K = mtxA.n;

  // void cblas_sgemm(
  //   CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
  //   const CBLAS_INT M, const CBLAS_INT N, const CBLAS_INT K,
  //   const float alpha,
  //   const float *A, const CBLAS_INT lda, const float *B, const CBLAS_INT ldb,
  //   const float beta,
  //   float *C, const CBLAS_INT ldc
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
    M, N,
    -1.0,
    mtxC.data, mtxC.ld, accu.data, accu.ld
  );

  return f_norm(accu);
}

int main( int argc, char** argv ) {
  if ( argc != 4 ) {
    std::cout << "Usage: " << argv[0] << " <M> <K> <N>" << "\n"
      << "\n"
      << "Computes the product of random matrices (constant seed):\n"
      << "  C = A * B" << "\n"
      << "  where" << "\n"
      << "    A : M[lda] x K" << "\n"
      << "    B : K[ldb] x N" << "\n"
      << "    C : M[ldc] x N" << "\n"
      << std::endl;
      exit(EXIT_FAILURE);
  }

  size_t const M = std::stoll(argv[1]);
  size_t const K = std::stoll(argv[2]);
  size_t const N = std::stoll(argv[3]);
  size_t const lda = M;
  size_t const ldb = K;
  size_t const ldc = M;

  int64_t seed_value = 0;

  DenseMatrix<float> mtxA(lda*K);
  mtxA.ld = lda;
  mtxA.m = M;
  mtxA.n = K;
  fill_random<float>(mtxA.data, mtxA.nnz, std::seed_seq{seed_value});

  seed_value++;

  DenseMatrix<float> mtxB(ldb*N);
  mtxB.ld = ldb;
  mtxB.m = K;
  mtxB.n = N;
  fill_random<float>(mtxB.data, mtxB.nnz, std::seed_seq{seed_value});

  DenseMatrix<float> mtxC(ldc*N);
  mtxC.ld = ldc;
  mtxC.m = M;
  mtxC.n = N;
  std::fill_n(mtxC.data, mtxC.nnz, 0.0);

  float const alpha = 1.0;
  float const beta = 0.0;

  float duration = 0.0;
  gpu_GEMM(alpha, mtxA, mtxB, beta, mtxC, gpu_GEMM_naive, duration);
  float error = gemm_error(alpha, mtxA, mtxB, beta, mtxC);

  std::cout << "GEMM naive\n"
            << "  - Error (Frobenius norm): " << error << "\n"
            << "  - Duration [ms]: " << duration  << "\n";

  duration = 0.0;
  gpu_GEMM(alpha, mtxA, mtxB, beta, mtxC, gpu_GEMM_coalesce, duration);
  error = gemm_error(alpha, mtxA, mtxB, beta, mtxC);

  std::cout << "GEMM (anti) coalesce\n"
            << "  - Error (Frobenius norm): " << error << "\n"
            << "  - Duration [ms]: " << duration  << "\n";

  return EXIT_SUCCESS;
}
