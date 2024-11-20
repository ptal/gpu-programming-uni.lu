# Matrix multiplication

In this exercise you are tasked with implementing the tiled matrix multiplication algorithm. All the functions provided follow the BLAS notation, so make sure you familiarize yourself with the [interface of BLAS](https://www.netlib.org/blas/).

The skeleton code provided uses randomly generated matrices to test your code. You just need to provide the matrix dimension as input. We will test you code on a set of input cases within another executable.

## Tiled matrix multiplication

Tiled matrix multiplication takes advantage of the cache memory in GPUs and other systems to reduce expensive fetches from the main memory. In GPUs you have direct control of the cache in the form of shared memory; variables stored in the shared memory of kernel functions are accessed much faster that variables in the main memory.

In the skeleton code of your project you can find the `gpu_GEMM_naive` kernel that implements the sGEMM algorithm of BLAS [1], without support for transposition operation, using direct access to the main memory. You objective is to implement the tiled matrix multiplication where you fetch tiles of matrix A and matrix B in the shared memory to reduce the number of individual accesses to the main memory. Remember, accessing consecutive regions in the main memory in a single operation is faster than accessing the same regions in individual operations [2].

```
void gpu_GEMM_tiling(
  size_t const M, size_t const N, size_t const K,
  T const alpha,
  T const* const A, size_t const ldA, T const* const B, size_t const ldB,
  T const beta,
  T* const C, size_t const ldC
) {
  // Assume Fortran like row major allocation: x(row)(column) = x[row + ld*column]
  __shared__ float tile_A(BLOCK_SIDE)(BLOCK_SIDE);
  __shared__ float tile_B(BLOCK_SIDE)(BLOCK_SIDE);

  for ( t = 0; t < ceil(M, BLOCK_SIDE); t++ ) {
    # for each thread (ti,tj) in block // Note the 2d thread indexing
    tile_A(ti, tj) <- A[ appropriate index ]
    tile_B(ti, tj) <- B[ appropriate index ]
  
    ac = 0
    for ( l = 0; l < BLOCK_SIDE; l++ ) {
      ac += tile_A(ti, l) * tile_B(i, tj)
    }
  }

  C[appropriate index] <- alpha * ac + beta * C[appropriate index]
}
```

This outline describes the high level operations performed by the code. Note that `<-` corresponds to accesses to the main memory. In your implementation you should pay attention in the following details:

- Handle edge cases, literally cases where the tile overflows at the end of the matrix.
- Perform any index arithmetic required to access the correct indices of matrices A and B.
- Make sure you understand and use correctly the interface of BLAS.

## Objectives and rules

1. Implement the `gpu_GEMM_tiling` algorithm and make sure that the error norm reported is zero. [12 points]
2. The conventional BLAS algorithm is used for evaluating the results of your implementation. The CPU is lather slow in testing larger cases. Replace the call to BLAS with a call to cuBLAS [3] in function `gemm_error`. [6 points]
3. Which algorithm executes faster, the `gpu_GEMM_tiling` or `gpu_GEMM_naive`? Can you improve the performance of the `gpu_GEMM_tiling` algorithm by improving the computational intensity [2]? [2 points]

Send the code and report in "name.surname.gemm.zip" by email with the subject "[MHPC][GP] GEMM project" to [Georgios Kafanas <georgios.kafanas@uni.lu>](mailto:georgios.kafanas@uni.lu) before the course begins at 13h59 on December 4.

## _Resources_

1. [The webpage of BLAS](https://www.netlib.org/blas/)
2. Section 1.5 of "Matrix Computations"
3. [cuBLAS Level-3 Function Reference](https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference)

You can find a copy of the 1st chapter of "Matrix Computations" in the documentation (`doc`) directory of the project.
