#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void sample_harmonic_signal(
  double const alpha, double const omega, double const phi,
  int const n, double* const v )
{
  for ( int t = 0; t < n; ++t ) {
    v[t] = alpha*sin( t*omega + phi );
  }
}

void copy_coo_to_dns( int const nz, int const ld, int const ncols,
  int const* const row, int const* const col, double const* const a,
  double* const val )
{
  int const n_entries = ld*ncols;

  for ( int i = 0; i < n_entries; ++i ) {
    val[ i ] = 0.0;
  }

  for ( int n = 0; n < nz; ++n ) {
    int const idx = row[n] + col[n]*ld;
    val[ idx ] = a[n];
  }
}

int get_ld( int const entries, int const type_cache_alignment )
{
  int cache_lines = entries / type_cache_alignment;
  
  if ( entries % type_cache_alignment != 0 ) {
    cache_lines += 1;
  }

  int const aligned_entries = cache_lines * type_cache_alignment;

  return aligned_entries;
}

void print_matrix_struct( FILE* const f, matrix const* const a )
{
  if ( a == NULL ) return;

  print_matrix( f, a->ld, a->nrows, a->ncols, a->data );
}

void print_vector_struct( FILE* const f, vector const* const v )
{
  if ( v == NULL ) return;

  print_vector( f, v->nrows, v->inc, v->data );
}

void print_matrix( FILE* const f,
  int const ld, int const nrows, int const ncols, double const* const a )
{
  for ( int row = 0; row < nrows; ++row ) {
    for ( int col = 0; col < ncols; ++col ) {
      fprintf(f, "%#.3lg ", a[row + col*ld]);
    }
    fprintf(f, "\n");
  }
}

void print_vector( FILE* const f,
  int const nrows, int const inc, double const* const v )
{
  for ( int n = 0; n < nrows; ++n ) {
    fprintf(f, "%lg\n", v[ n * inc ]);
  }
}
