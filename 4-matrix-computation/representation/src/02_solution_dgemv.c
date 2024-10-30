#include <stdlib.h>
#include <stdio.h>

#include <cblas.h>

#include "utils.h"

static void call_dgemv( matrix const* const A, vector const* const x, vector* const y )
{
  double const alpha = 1.0;
  double const beta = 0.0;
  cblas_dgemv( CblasColMajor,
    CblasNoTrans, A->nrows, A->ncols,
    alpha,
    A->data, A->ld,
    x->data, x->inc,
    beta,
    y->data, y->inc
 );
}

static void get_A( matrix* const A )
{
  if ( A == NULL ) return;

  A->nrows = 2;
  A->ncols = 3;
  A->ld = 2;
  A->size = A->ld * A->ncols;

  A->data = (double*) malloc( A->size * sizeof(double) );
  if ( A->data == NULL ) {
    fprintf( stderr, "Memory allocation for matrix A failed.\n" );
    exit( EXIT_FAILURE );
  }

  A->data[0] = 1.0;
  A->data[1] = 2.0;
  A->data[2] = 3.0;
  A->data[3] = 4.0;
  A->data[4] = 5.0;
  A->data[5] = 6.0;
}

static void get_x( vector* const x )
{
  if ( x == NULL ) return;

  x->nrows = 3;
  x->inc = 1;
  x->size = x->nrows * x->inc;

  x->data = (double*) malloc( x->size * sizeof(double) );
  if ( x->data == NULL ) {
    fprintf( stderr, "Memory allocation for vector x failed.\n" );
    exit( EXIT_FAILURE );
  }

  x->data[0] = 1.0;
  x->data[1] = 3.0;
  x->data[2] = 2.0;
}

static void get_y( vector* const y )
{
  if ( y == NULL ) return;

  y->nrows = 2;
  y->inc = 1;
  y->size = y->nrows * y->inc;

  y->data = (double*) malloc( y->size * sizeof(double) );
  if ( y->data == NULL ) {
    fprintf( stderr, "Memory allocation for vector y failed.\n" );
    exit( EXIT_FAILURE );
  }

  y->data[0] = 0.0;
  y->data[1] = 0.0;
}

static void test_dgemv()
{

  matrix A;
  vector x;
  vector y;

  get_A(&A);
  get_x(&x);
  get_y(&y);

  call_dgemv( &A, &x, &y );

  printf( "*** Result vector ***\n" );
  print_vector_struct( stdout, &y ); 

  free(A.data);
  free(x.data);
  free(y.data);
}

int main( int argc, char** argv )
{
  if ( argc != 1 ) {
    fprintf( stderr, "Usage: %s\n", argv[0] );
    return EXIT_FAILURE;
  }

  test_dgemv();

  return EXIT_SUCCESS;
}
