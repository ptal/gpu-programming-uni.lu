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

static void get_M( matrix* const A, int const m, int const n )
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

static void get_v( vector* const x, int const n )
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

static void get_aligned_M( matrix* const A, int const m, int const n )
{
  if ( A == NULL ) return;

  A->nrows = m;
  A->ncols = n;
  // Use get_ld to have memory aligned columns
  A->ld = A->nrows;
  A->size = A->ld * A->ncols;

  // Use aligned_alloc to get aligned memory
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

static void get_aligned_v( vector* const x, int const n )
{
  if ( x == NULL ) return;

  x->nrows = n;
  x->inc = 1;
  x->size = x->nrows * x->inc;

  // Use aligned_alloc to get aligned memory
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

static void test_alignment_effect( int const m, int const n )
{
  printf( "*** Testing DDGEMV performance ***\n" );

  double const alpha = 1.0;
  double const beta = 2.0;

  matrix A;
  vector x;
  vector y;

  get_M( &A, m, n );
  get_v( &x, n );
  get_v( &y, m );

  clock_t start = clock();
  cblas_dgemv( CblasColMajor,
    CblasNoTrans, A.nrows, A.ncols,
    alpha,
    A.data, A.ld,
    x.data, x.inc,
    beta,
    y.data, y.inc);
  clock_t diff = clock() - start;
  double duration = (double) diff / CLOCKS_PER_SEC;

  free(A.data);
  free(x.data);
  free(y.data);

  printf("non-aligned duration = %f\n", duration ); 

  get_aligned_M( &A, m, n );
  get_aligned_v( &x, n );
  get_aligned_v( &y, m );

  start = clock();
  cblas_dgemv( CblasColMajor,
    CblasNoTrans, A.nrows, A.ncols,
    alpha,
    A.data, A.ld,
    x.data, x.inc,
    beta,
    y.data, y.inc);
  diff = clock() - start;
  duration = (double) diff / CLOCKS_PER_SEC;

  free(A.data);
  free(x.data);
  free(y.data);

  printf("aligned duration = %f\n", duration ); 
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

  int m, n;
  int err = parse_size_arg( argv[1], &m );
  if ( err ) return err;
  err = parse_size_arg( argv[2], &n );
  if ( err ) return err;

  load_blas();
  test_alignment_effect( m, n );

  return EXIT_SUCCESS;
}
