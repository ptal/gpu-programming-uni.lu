###
# Add external libraries

set( LOCAL_OPT_DIR "$ENV{HOME}/.local" )

add_library( matrix-market SHARED IMPORTED )
set_target_properties( matrix-market PROPERTIES
  IMPORTED_LOCATION "${LOCAL_OPT_DIR}/matrix-market/lib64/libmatrixmarket.so"
  INTERFACE_INCLUDE_DIRECTORIES "${LOCAL_OPT_DIR}/matrix-market/include"
  POSITION_INDEPENDENT_CODE ON
)

add_library( blas SHARED IMPORTED )
set_target_properties( blas PROPERTIES
  IMPORTED_LOCATION "${LOCAL_OPT_DIR}/netlib/lib64/libblas.so"
  INTERFACE_INCLUDE_DIRECTORIES "${LOCAL_OPT_DIR}/netlib/include"
  POSITION_INDEPENDENT_CODE ON
)
add_library( cblas SHARED IMPORTED )
set_target_properties( cblas PROPERTIES
  IMPORTED_LOCATION "${LOCAL_OPT_DIR}/netlib/lib64/libcblas.so"
  INTERFACE_INCLUDE_DIRECTORIES "${LOCAL_OPT_DIR}/netlib/include"
  POSITION_INDEPENDENT_CODE ON
)

###
# Create utility library

set(UTILS_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/utils.c")
add_library(utils STATIC "${UTILS_SRC_FILES}")
target_link_libraries( utils PRIVATE m )

###
# Create example executables

# 00. EXAMPLE
set( 00_EXAMPLE_DDOT_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/00_example_ddot.c")
add_executable(00_example_ddot "${00_EXAMPLE_DDOT_SRC_FILES}")
target_include_directories(00_example_ddot PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 00_example_ddot PRIVATE utils cblas blas )

# 01. EXCERCISE
set( 01_EXERCISE_DAXPY_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/01_exercise_daxpy.c")
add_executable(01_exercise_daxpy "${01_EXERCISE_DAXPY_SRC_FILES}")
target_include_directories(01_exercise_daxpy PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 01_exercise_daxpy PRIVATE utils cblas blas )

set( 01_SOLUTION_DAXPY_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/01_solution_daxpy.c")
add_executable(01_solution_daxpy "${01_SOLUTION_DAXPY_SRC_FILES}")
target_include_directories(01_solution_daxpy PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 01_solution_daxpy PRIVATE utils cblas blas )

# 02. EXCERCISE
set( 02_EXERCISE_DGEMV_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/02_exercise_dgemv.c")
add_executable(02_exercise_dgemv "${02_EXERCISE_DGEMV_SRC_FILES}")
target_include_directories(02_exercise_dgemv PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 02_exercise_dgemv PRIVATE utils cblas blas )

set( 02_SOLUTION_DGEMV_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/02_solution_dgemv.c")
add_executable(02_solution_dgemv "${02_SOLUTION_DGEMV_SRC_FILES}")
target_include_directories(02_solution_dgemv PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 02_solution_dgemv PRIVATE utils cblas blas )

# 03. EXCERCISE
set( 03_EXERCISE_DGEMM_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/03_exercise_dgemm.c")
add_executable(03_exercise_dgemm "${03_EXERCISE_DGEMM_SRC_FILES}")
target_include_directories(03_exercise_dgemm PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 03_exercise_dgemm PRIVATE utils cblas blas )

set( 03_SOLUTION_DGEMM_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/03_solution_dgemm.c")
add_executable(03_solution_dgemm "${03_SOLUTION_DGEMM_SRC_FILES}")
target_include_directories(03_solution_dgemm PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 03_solution_dgemm PRIVATE utils cblas blas )

# 04. EXCERCISE
set( 04_EXERCISE_CACHE_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/04_exercise_cache.c")
add_executable(04_exercise_cache "${04_EXERCISE_CACHE_SRC_FILES}")
target_include_directories(04_exercise_cache PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 04_exercise_cache PRIVATE utils cblas blas )

set( 04_SOLUTION_CACHE_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/04_solution_cache.c")
add_executable(04_solution_cache "${04_SOLUTION_CACHE_SRC_FILES}")
target_include_directories(04_solution_cache PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 04_solution_cache PRIVATE utils cblas blas )

# 05. EXCERCISE
set( 05_EXERCISE_ALIGNMENT_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/05_exercise_alignment.c")
add_executable(05_exercise_alignment "${05_EXERCISE_ALIGNMENT_SRC_FILES}")
target_include_directories(05_exercise_alignment PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 05_exercise_alignment PRIVATE utils cblas blas )

set( 05_SOLUTION_ALIGNMENT_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/05_solution_alignment.c")
add_executable(05_solution_alignment "${05_SOLUTION_ALIGNMENT_SRC_FILES}")
target_include_directories(05_solution_alignment PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( 05_solution_alignment PRIVATE utils cblas blas )


# DEMOS

set(READ_MATRIX_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_FILES}/read_matrix.c")
add_executable(read_matrix "${READ_MATRIX_SRC_FILES}")
target_include_directories(read_matrix PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( read_matrix PRIVATE matrix-market )

set(READ_AND_EXECUTE_SRC_FILES
  "${CMAKE_SOURCE_DIR}/${SRC_FILES}/read_and_execute.c"
)
add_executable(read_and_execute "${READ_AND_EXECUTE_SRC_FILES}")
target_include_directories(read_and_execute PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( read_and_execute PRIVATE matrix-market utils cblas blas )

set(EXAMPLE_EXECUTE_SRC_FILES
  "${CMAKE_SOURCE_DIR}/${SRC_FILES}/example_execute.c"
)
add_executable(example_execute "${EXAMPLE_EXECUTE_SRC_FILES}")
target_include_directories(example_execute PRIVATE
  "${CMAKE_SOURCE_DIR}/${INCLUDE_FILES}"
)
target_link_libraries( example_execute PRIVATE cblas blas utils )

##target_link_directories( read_matrix PRIVATE "$ENV{HOME}/opt/matrix-market/lib")
#find_library( MM NAME matrixmarket 
#  HINTS "$ENV{HOME}/opt/matrix-market/lib/"
#)
#set_target_properties( read_matrix PROPERTIES
#  INCLUDE_DIRECTORIES "$ENV{HOME}/opt/matrix-market/include"
#)
