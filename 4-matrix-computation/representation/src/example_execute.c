#include <stdlib.h>
#include <stdio.h>

#include <cblas.h>

#include "utils.h"

int main( int argc, char** argv )
{
  int const M = 2;
  int const N = 2;
  int const ld = get_ld( M, AVX512ALIGN / sizeof(double) );
  double* A = aligned_alloc( AVX512ALIGN, ld*N );

  printf("ld = %d\n", ld);
  A[0 + 0*ld] = 1.0;
  A[1 + 0*ld] = 0.0;
  A[0 + 1*ld] = 0.0;
  A[1 + 1*ld] = 2.0;

  double x[] = {1.0, 1.0};
  int const incx = 1;
  double y[] = {0.0, 0.0};
  int const incy = 1; 
  double const alpha = 1.0;
  double const beta = 0.0;

  printf("y = \n");
  print_vector( stdout, N, incy, y );

  cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N,
    alpha, A, ld,
    x, incx, beta,
    y, incy );

  printf("A = \n");
  print_matrix( stdout, ld, M, N, A );

  printf("y = \n");
  print_vector( stdout, N, incy, y );

  free(A);

  return EXIT_SUCCESS;
}
