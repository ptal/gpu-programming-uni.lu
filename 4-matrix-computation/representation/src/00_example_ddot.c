#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cblas.h>

#include "utils.h"

static void test_edge_case_ddot()
{
  int const n = 2;
  double const x[] = { -1.0, 1.0, 1000.0 };
  double const y[] = {  1.0, 2.0 };
  
  int incx = 1;
  int incy = 1;
  double const ddot_incx1 = cblas_ddot( n, x, incx, y, incy );

  incx = 2;
  incy = 1;
  double const ddot_incx2 = cblas_ddot( n, x, incx, y, incy );

  printf( "*** Test increment edge cases ***\n" );
  printf( "Case [ incx = 1 ] -> %f\n", ddot_incx1 );
  printf( "Case [ incx = 2 ] -> %f\n", ddot_incx2 );
}

static void test_performance_ddot( int const n, int const incx )
{
  double* const x = (double*) malloc( (n * incx) * sizeof(double) );
  double* const y = (double*) malloc( n * sizeof(double) );

  double const phi = 0.0;
  double omega = 0.1;
  double alpha = 0.5;

  sample_harmonic_signal( alpha, omega, phi, n, x );
  
  omega = 0.5;
  alpha = 0.1;
  sample_harmonic_signal( alpha, omega, phi, n, y );

  int const incy = 1;

  clock_t start = clock();
  double const ddot = cblas_ddot(n,
    x, incx,
    y, incy
  );
  clock_t diff = clock() - start;
  double const duration = (double) diff / CLOCKS_PER_SEC;

  printf( "*** Generic performance test ***\n" );
  printf( "The dot-product of 2 vectors [ n=%d, incx=%d, incy=%d ] was evaluated to: %f\n",
    n, incx, incy, ddot
  );
  printf( "Time required for ddot: %f\n", duration );

  free(y);
  free(x);
}

static int parse_size_arg( char* argvi, int* val )
{
  int const base = 10;
  char* end;
  *val = strtol( argvi, &end, base );
  if ( end == argvi ) {
    fprintf( stderr, "Invalid vector size.\n" );
    return EXIT_FAILURE;
  } else if ( *end != '\0') {
    fprintf( stderr, "Invalid input: %s\n", argvi );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main( int argc, char** argv )
{
  if ( argc != 3 ) {
    fprintf( stderr, "Usage: %s <test vector size> <test vector increment>\n", argv[0] );
    return EXIT_FAILURE;
  }

  int n, incx;
  int err = parse_size_arg( argv[1], &n );
  if ( err ) return err;
  err = parse_size_arg( argv[2], &incx );
  if ( err ) return err;

  test_edge_case_ddot();
  test_performance_ddot( n, incx );

  return EXIT_SUCCESS;
}
