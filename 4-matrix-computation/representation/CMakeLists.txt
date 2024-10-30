cmake_minimum_required(VERSION 3.15)

# Add directory with utility functions and modules in the serach path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include basic utility fuinctions
include(utilities)

project(market-matrix-exchange-format VERSION 1.0)

set_build_type_parameters()
set_output_dir_structure()

# Add dependencies explictly, otherwise you may link with Netlib CBLAS and system BLAS
add_link_options("LINKER:--no-as-needed")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src")
