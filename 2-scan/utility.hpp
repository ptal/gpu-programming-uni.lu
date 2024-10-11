// Copyright 2023 Pierre Talbot

#include <cstdlib>
#include <cstdio>

#pragma once

inline static void handle_error( cudaError_t err,
                                 const char *file,
                                 int line) {
  if ( err != cudaSuccess ) {
    printf( "%s:%d CUDA runtime error %s\n", file, line, cudaGetErrorString( err ) );
    exit( EXIT_FAILURE );
  }
}

#define try_cuda( result ) handle_error( (result), __FILE__, __LINE__ )
