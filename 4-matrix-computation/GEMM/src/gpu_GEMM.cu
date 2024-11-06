#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <random>
#include <cmath>

#include <cblas.h>

#include "utility.hpp"
#include "datatypes.hpp"

#define N_BLOCK_SIZE 16
#define M_BLOCK_SIZE 16

template<typename T>
using gemm_fcn = void (*)(
  T const* const A, T const* const B, T* const C,
  size_t const M, size_t const N, size_t const K,
  size_t const ldA, size_t const ldB, size_t const ldC
);

template<typename T>
void gpu_GEMM(
  DenseMatrix<T> const& mtxA, DenseMatrix<T> const& mtxB, DenseMatrix<T>& mtxC,
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

  size_t m_grid_size = mtxC.m/M_BLOCK_SIZE;
  if ( mtxC.m % M_BLOCK_SIZE != 0 ) m_grid_size++;
  size_t n_grid_size = mtxC.n/N_BLOCK_SIZE;
  if ( mtxC.n % N_BLOCK_SIZE != 0 ) n_grid_size++;

  dim3 threadsPerBlock(M_BLOCK_SIZE, N_BLOCK_SIZE);
  dim3 blocksPerGrid(m_grid_size, n_grid_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  try_CUDA( cudaMemcpy(A, mtxA.data, sizeof(T) * mtxA.nnz, cudaMemcpyHostToDevice) );
  try_CUDA( cudaMemcpy(B, mtxB.data, sizeof(T) * mtxB.nnz, cudaMemcpyHostToDevice) );

  gemm<<<blocksPerGrid, threadsPerBlock>>>(
    A, B, C,
    mtxA.m, mtxA.n, mtxB.n,
    mtxA.ld, mtxB.ld, mtxC.ld
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
  T const* const A, T const* const B, T* const C,
  size_t const M, size_t const N, size_t const K,
  size_t const ldA, size_t const ldB, size_t const ldC
) {
  size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t const j = blockIdx.y * blockDim.y + threadIdx.y;

  if ( i >= M || j >= K ) return;

  T c_ij = static_cast<T>(0);

  for ( size_t l = 0; l < N; l++ ) {
    T const a_il = A[i + ldA*l];
    T const b_lj = B[l + ldB*j];
    c_ij += a_il * b_lj;
  }

  C[i + ldC*j] = c_ij;
}

template<typename T>
__global__ void gpu_GEMM_coalesce(
  T const* const A, T const* const B, T* const C,
  size_t const M, size_t const N, size_t const K,
  size_t const ldA, size_t const ldB, size_t const ldC
) {

  size_t const i = blockIdx.x * blockDim.x + threadIdx.x / blockDim.x + threadIdx.y;
  size_t const j = blockIdx.y * blockDim.y + threadIdx.y / blockDim.y + threadIdx.x;

  if ( i >= M || j >= K ) return;

  T c_ij = static_cast<T>(0);

  for ( size_t l = 0; l < N; l++ ) {
    T const a_il = A[i + ldA*l];
    T const b_lj = B[l + ldB*j];
    c_ij += a_il * b_lj;
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

int main( int argc, char** argv ) {
  if ( argc != 4 ) {
    std::cout << "Usage: " << argv[0]
      << "<M> <N> <K>" << "\n"
      << "\n"
      << "Computes the product of random matrices (constant seed):\n"
      << "  C = A * B" << "\n"
      << "  where" << "\n"
      << "    A : M[lda] x N" << "\n"
      << "    B : N[ldb] x K" << "\n"
      << "    C : M[ldc] x K" << "\n"
      << std::endl;
      exit(EXIT_FAILURE);
  }

  size_t const M = std::stoll(argv[1]);
  size_t const N = std::stoll(argv[2]);
  size_t const K = std::stoll(argv[3]);
  size_t const lda = M;
  size_t const ldb = N;
  size_t const ldc = M;

  int64_t seed_value = 0;

  DenseMatrix<float> mtxA(lda*N);
  mtxA.ld = lda;
  mtxA.m = M;
  mtxA.n = N;
  fill_random<float>(mtxA.data, mtxA.nnz, std::seed_seq{seed_value});

  seed_value++;

  DenseMatrix<float> mtxB(ldb*K);
  mtxB.ld = ldb;
  mtxB.m = N;
  mtxB.n = K;
  fill_random<float>(mtxB.data, mtxB.nnz, std::seed_seq{seed_value});

  DenseMatrix<float> mtxC(ldc*K);
  mtxC.ld = ldc;
  mtxC.m = M;
  mtxC.n = K;
  std::fill_n(mtxC.data, mtxC.nnz, 0.0);

  float duration = 0.0;
  gpu_GEMM(mtxA, mtxB, mtxC, gpu_GEMM_naive, duration);

  std::cout << "GEMM naive\n"
            << "  - Error (Frobenius norm): "
            << gemm_error(mtxA, mtxB, mtxC) << "\n"
            << "  - Duration [ms]: " << duration  << "\n";

  duration = 0.0;
  gpu_GEMM(mtxA, mtxB, mtxC, gpu_GEMM_coalesce, duration);

  std::cout << "GEMM (anti) coalesce\n"
            << "  - Error (Frobenius norm): "
            << gemm_error(mtxA, mtxB, mtxC)  << "\n"
            << "  - Duration [ms]: " << duration  << "\n";

  return EXIT_SUCCESS;
}
