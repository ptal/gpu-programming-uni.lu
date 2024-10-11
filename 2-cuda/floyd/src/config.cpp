// Copyright 2023 Pierre Talbot

#include "config.hpp"
#include <iostream>

void usage_and_exit(const std::string& program_name) {
  std::cout << "usage: " << program_name << " <matrix size> <threads-per-block> <num-blocks>" << std::endl;
  exit(EXIT_FAILURE);
}
