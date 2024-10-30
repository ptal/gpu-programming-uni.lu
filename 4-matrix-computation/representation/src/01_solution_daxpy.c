#include <stdlib.h>
#include <stdio.h>

#include <cblas.h>

#include "utils.h"

static void call_daxpy( vector* const v, double const alpha )
{
  cblas_daxpy( v->nrows, alpha - 1.0, v->data, v->inc, v->data, v->inc );
}

static void test_daxpy()
{
  double const alpha = 2.0;
  
  int const nrows = 2;
  int const inc = 1;
  int const size = nrows*inc;
  double* const data = (double*) malloc( size * sizeof(double) );
  if ( data == NULL ) {
    fprintf( stderr, "Memory allocation failed.\n" );
    exit( EXIT_FAILURE );
  }

  data[0] = 1.0;
  data[1] = 2.0;

  vector v;
  v.data = data;
  v.inc = inc;
  v.nrows = nrows;
  v.size = size;

  printf( "*** Original vector X ***\n" );
  print_vector_struct( stdout, &v ); 
  call_daxpy( &v, alpha );
  printf( "*** %f*X ***\n", alpha );
  print_vector_struct( stdout, &v ); 

  free(data);
}

int main( int argc, char** argv )
{
  if ( argc != 1 ) {
    fprintf( stderr, "Usage: %s\n", argv[0] );
    return EXIT_FAILURE;
  }

  test_daxpy();

  return EXIT_SUCCESS;
}
