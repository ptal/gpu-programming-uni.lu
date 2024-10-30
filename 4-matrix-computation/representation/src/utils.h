#include <stdlib.h>
#include <stdio.h>

// AVX-512 used 64byte alignment
#define AVX512ALIGN 64

struct _vector {
  /* Data structure for dense real matrices */
  double* data;
  int nrows; // Number of rows
  int inc;   // Increment
  int size;  // Maximun capacity of the space allocated for v 
};
typedef struct _vector vector;

struct _matrix {
  /* Data structure for dense real matrices */
  double* data;
  int nrows; // Number of rows
  int ncols; // Number of columns
  int ld;    // Leading dimension
  int size;  // Maximun capacity of the space allocated for a
};
typedef struct _matrix matrix;

void sample_harmonic_signal(
  double const alpha, double const omega, double const phi,
  int const n, double* const v );

void copy_coo_to_dns( int const nz, int const ld, int const ncols,
  int const* const row, int const* const col, double const* const a,
  double* const val );

int get_ld( int const entries, int const type_cache_alignment );

void print_matrix_struct( FILE* const f, matrix const* const a );

void print_vector_struct( FILE* const f, vector const* const v );

void print_matrix( FILE* const f,
  int const ld, int const nrows, int const ncols, double const* const a );

void print_vector( FILE* const f,
  int const nrows, int const inc, double const* const v );
