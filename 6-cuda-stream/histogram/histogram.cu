#include <vector>
#include <iostream>
#include <thread>
#include <span>
#include <random>
#include <algorithm>

#define MAX_VALUE 10000

class DataReader {
  std::mt19937 r;
  std::uniform_int_distribution<int> dist;

public:
  /** For replicability of the experiments, we default-initialize `m`.
   * We draw numbers between 0 and MAX_VALUE. */
  DataReader(): dist(0, MAX_VALUE) {}

  /** Fill `data` with `data.size()` numbers greater or equal to 0. */
  void read_data(std::vector<int>& data) {
    std::generate(data.begin(), data.end(), [&](){return dist(r);});
  }
};

void check_equal_vector(const std::vector<int>& a, const std::vector<int>& b) {
  if(a.size() != b.size()) {
    std::cerr << "Size of arrays differs..." << std::endl;
  }
  else {
    for(size_t i = 0; i < a.size(); ++i) {
      if(a[i] != b[i]) {
        std::cerr << "Found an error: " << a[i] << " != " << b[i] << " at index " << i << std::endl;
        exit(1);
      }
    }
  }
}

std::vector<int> histogram_cpu_naive(size_t n, int k) {
  DataReader reader;
  std::vector<int> data(n, 0);
  std::vector<int> histogram(k, 0);
  reader.read_data(data);
  for(size_t i = 0; i < data.size(); ++i) {
    ++histogram[data[i] / (MAX_VALUE/k)];
  }
  return std::move(histogram);
}

int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "usage: " << argv[0] << " <n> <k>" << std::endl;
    std::cout << "  n: the number of integers in the sequence.\n";
    std::cout << "  k: the number of bins of the histogram.\n";
    exit(1);
  }
  size_t n = std::stoll(argv[1]);
  size_t k = std::stoll(argv[2]);
  /** The time includes the memory allocation. */
  auto start = std::chrono::steady_clock::now();
  std::vector<int> histogram = histogram_cpu_naive(n, k);
  auto end = std::chrono::steady_clock::now();
  std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
  for(int i = 0; i < histogram.size(); ++i) {
    std::cout << i << ' ' << histogram[i] << std::endl; // std::string(histogram[i] / (n/100), '*') << '\n';
  }
  return 0;
}
