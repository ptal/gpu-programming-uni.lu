#include <stdlib.h>
#include <stdio.h>

#include <cblas.h>
#include "mmio.h"

#include "utils.h"

int main( int argc, char** argv )
{
  if ( argc != 2 ) {
    printf("Usage: %s <matrix file name>\n", argv[0]);
    return EXIT_SUCCESS;
  }

  FILE* f = (FILE*) fopen(argv[1], "r");
  if ( f == NULL ) {
    fprintf(stderr, "fopen() failed in file %s at line # %d", __FILE__, __LINE__);
    return EXIT_FAILURE;
  }

  MM_typecode type;
  int res = mm_read_banner(f, &type);
  if ( res ) {
    fprintf(stderr, "Failed to read the matrix banner in file %s at line # %d", __FILE__, __LINE__);
    return EXIT_FAILURE;
  }

  if ( ! ( mm_is_matrix(type) && mm_is_sparse(type) && mm_is_real(type) ) ) {
    printf("This example expects a real sparse matrix!\n");
    return EXIT_SUCCESS;
  }

  int M, N, nz;
  res = mm_read_mtx_crd_size(f, &M, &N, &nz);
  if ( res ) {
    fprintf(stderr, "Failed to read the matrix size data in file %s at line # %d", __FILE__, __LINE__);
    return EXIT_FAILURE;
  }

  int* I = (int*) aligned_alloc( AVX512ALIGN, nz*sizeof(int) );
  int* J = (int*) aligned_alloc( AVX512ALIGN, nz*sizeof(int) );
  double* val = (double*) aligned_alloc( AVX512ALIGN, nz*sizeof(double) );

  res = mm_read_mtx_crd_real_data( f, nz, I, J, val );
  if ( res ) {
    fprintf(stderr, "Failed to read the matrix data in file %s at line # %d", __FILE__, __LINE__);
    return EXIT_FAILURE;
  }

  res = fclose(f);
  if ( res ) {
    fprintf(stderr, "Failed to close the file (see 'errno' more details) in file %s at line # %d", __FILE__, __LINE__);
    return EXIT_FAILURE;
  }

  int const ld = get_ld( M, AVX512ALIGN / sizeof(double) );
  printf("ld = %d\n", ld);
  printf("ld*N = %d\n", ld*N);
  double* A = (double*) aligned_alloc( AVX512ALIGN, ld*N*sizeof(double) );

  copy_coo_to_dns( nz, ld, N,
    I, J, val,
    A );

  free(I);
  free(J);
  free(val);

  double* x = (double*) aligned_alloc( AVX512ALIGN, N*sizeof( double ) );
  int const incx = 1;
  double* y = (double*) aligned_alloc( AVX512ALIGN, N*sizeof( double ) );
  int const incy = 1;

  for ( int i = 0; i < N; ++i ) {
    x[i] = 0.0;
    y[i] = 0.0;
  }
  x[0] = 1.0;

  double const alpha = 1.0;
  double const beta = 0.0;

  cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N,
    alpha, A, ld,
    x, incx, beta,
    y, incy );

  printf("y = \n");
  print_vector( stdout, N, incy, y );

  free( A );
  free( x );
  free( y );

  return EXIT_SUCCESS;
}
