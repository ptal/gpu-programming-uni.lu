#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cblas.h>

#include "utils.h"

static void load_blas()
{
  int const n = 2;
  double const x[] = { -1.0, 1.0 };
  double const y[] = {  1.0, 2.0 };
  
  int incx = 1;
  int incy = 1;
  double const ddot_incx1 = cblas_ddot( n, x, incx, y, incy );

  printf( "*** Loaded BLAS ***\n" );
}

static void ddot_multiplication( matrix const* A, vector const* x )
{
  if ( A == NULL || x == NULL ) return;

  vector y;

  y.nrows = A->nrows;
  y.inc = 1;
  y.size = y.nrows;
  
  y.data = (double*) malloc( (y.size) * sizeof(double) );
  if ( y.data == NULL ) {
    fprintf( stderr, "Memory allocation for matrix A failed.\n" );
    exit( EXIT_FAILURE );
  }

  clock_t start = clock();
  // Evaluate y <- Ax with xDOT
  clock_t diff = clock() - start;
  double const duration = (double) diff / CLOCKS_PER_SEC;

  free(y.data);

  printf("ddot duration = %f\n", duration ); 
}

static void daxpy_multiplication( matrix const* A, vector const* x )
{
  if ( A == NULL || x == NULL ) return;

  vector y;

  y.nrows = A->nrows;
  y.inc = 1;
  y.size = y.nrows;
  
  y.data = (double*) malloc( (y.size) * sizeof(double) );
  if ( y.data == NULL ) {
    fprintf( stderr, "Memory allocation for matrix A failed.\n" );
    exit( EXIT_FAILURE );
  }

  clock_t start = clock();
  // Evaluate y <- Ax with xAXPY
  clock_t diff = clock() - start;
  double const duration = (double) diff / CLOCKS_PER_SEC;

  free(y.data);

  printf("daxpy duration = %f\n", duration ); 
}

static void get_A( matrix* const A, int const m, int const n )
{
  if ( A == NULL ) return;

  A->nrows = m;
  A->ncols = n;
  A->ld = A->nrows;
  A->size = A->ld * A->ncols;

  A->data = (double*) malloc( (A->size) * sizeof(double) );
  if ( A->data == NULL ) {
    fprintf( stderr, "Memory allocation for matrix A failed.\n" );
    exit( EXIT_FAILURE );
  }

  double const alpha = 2.0;
  double const omega = 0.1;
  double const phi = 0.0;
  sample_harmonic_signal(
    alpha, omega, phi,
    A->size, A->data );
}

static void get_x( vector* const x, int const n )
{
  if ( x == NULL ) return;

  x->nrows = n;
  x->inc = 1;
  x->size = x->nrows * x->inc;

  x->data = (double*) malloc( (x->size) * sizeof(double) );
  if ( x->data == NULL ) {
    fprintf( stderr, "Memory allocation for matrix A failed.\n" );
    exit( EXIT_FAILURE );
  }

  double const alpha = 0.3;
  double const omega = 1;
  double const phi = 0.0;
  sample_harmonic_signal(
    alpha, omega, phi,
    x->size, x->data );
}

static void test_stride_effect( int const m, int const n )
{

  matrix A;
  vector x;

  get_A( &A, m, n );
  get_x( &x, n );

  printf( "*** Stride effect test ***\n" );
  ddot_multiplication( &A, &x );
  daxpy_multiplication( &A, &x );

  free(A.data);
  free(x.data);
}

static int parse_size_arg( char* argvi, int* val )
{
  int const base = 10;
  char* end;
  *val = strtol( argvi, &end, base );
  if ( end == argvi ) {
    fprintf( stderr, "Invalid size.\n" );
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
    fprintf( stderr, "Usage: %s <M> <N>\n", argv[0] );
    fprintf( stderr, "Where the size of the multiplied matrix is: M x N\n" );
    return EXIT_FAILURE;
  }

  load_blas();

  int m, n;
  int err = parse_size_arg( argv[1], &m );
  if ( err ) return err;
  err = parse_size_arg( argv[1], &n );
  if ( err ) return err;

  test_stride_effect( m, n );

  return EXIT_SUCCESS;
}
