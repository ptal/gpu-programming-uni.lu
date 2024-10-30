#include <stdlib.h>
#include <stdio.h>

#include <cblas.h>

#include "utils.h"

static void call_dgemm( matrix const* const A, matrix const* const B, matrix* const C )
{
  if ( A == NULL || B == NULL || C == NULL ) return;

  if ( A->ncols != B->nrows || C->nrows != A->nrows || C->ncols != B->ncols ) {
    fprintf( stderr, "Incompatible shapes.\n" );
    exit( EXIT_FAILURE );
  }

  int const m = A->nrows;
  int const k = A->ncols;
  int const n = B->ncols;
  int const lda = A->ld;
  int const ldb = B->ld;
  int const ldc = C->ld;

  double const alpha = 1.0;
  double const beta = 0.0;
  cblas_dgemm( CblasColMajor,
    CblasNoTrans, CblasNoTrans, 
    m, n, k,
    alpha,
    A->data, lda,
    B->data, ldb,
    beta,
    C->data, ldc
 );
}

static void get_A( matrix* const A )
{
  if ( A == NULL ) return;

  A->nrows = 2;
  A->ncols = 3;
  A->ld = 2;
  A->size = A->ld * A->ncols;

  A->data = (double*) malloc( (A->size) * sizeof(double) );
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

static void get_B( matrix* const B )
{
  if ( B == NULL ) return;

  B->nrows = 3;
  B->ncols = 2;
  B->ld = 3;
  B->size = B->ld * B->ncols;

  B->data = (double*) malloc( (B->size) * sizeof(double) );
  if ( B->data == NULL ) {
    fprintf( stderr, "Memory allocation for matrix B failed.\n" );
    exit( EXIT_FAILURE );
  }

  B->data[0] = 1.0;
  B->data[1] = 3.0;
  B->data[2] = 2.0;
  B->data[3] = 0.0;
  B->data[4] = 1.0;
  B->data[5] = 0.0;
}

static void get_C( matrix* const C )
{
  if ( C == NULL ) return;

  C->nrows = 2;
  C->ncols = 2;
  C->ld = 2;
  C->size = C->ld * C->ncols;

  C->data = (double*) malloc( (C->size) * sizeof(double) );
  if ( C->data == NULL ) {
    fprintf( stderr, "Memory allocation for matrix C failed.\n" );
    exit( EXIT_FAILURE );
  }

  C->data[0] = 0.0;
  C->data[1] = 0.0;
  C->data[2] = 0.0;
  C->data[3] = 0.0;
}

static void test_dgemm()
{

  matrix A;
  matrix B;
  matrix C;

  get_A(&A);
  get_B(&B);
  get_C(&C);

  call_dgemm( &A, &B, &C );

  printf( "*** A matrix ***\n" );
  print_matrix_struct( stdout, &A ); 
  printf( "*** B matrix ***\n" );
  print_matrix_struct( stdout, &B ); 
  printf( "*** C = A*B ***\n" );
  print_matrix_struct( stdout, &C ); 

  free(A.data);
  free(B.data);
  free(C.data);
}

int main( int argc, char** argv )
{
  if ( argc != 1 ) {
    fprintf( stderr, "Usage: %s\n", argv[0] );
    return EXIT_FAILURE;
  }

  test_dgemm();

  return EXIT_SUCCESS;
}
