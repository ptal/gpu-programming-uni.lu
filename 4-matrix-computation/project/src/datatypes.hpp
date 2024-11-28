#pragma once

#include <cstdlib>
#include <ostream>
#include <string>

template<typename T>
struct DenseMatrix {
  public:
    T* data;
    size_t m;
    size_t n;
    size_t ld;
    size_t const nnz;

  DenseMatrix(size_t const _nnz);
  ~DenseMatrix();
  void display(std::ostream& s) const;
};

template<typename T>
DenseMatrix<T>::DenseMatrix(size_t const _nnz) :
  data(nullptr),
  m(0),
  n(0),
  ld(0),
  nnz(_nnz)
{
  data = new T[_nnz];
}

template<typename T>
DenseMatrix<T>::~DenseMatrix() {
  if (data) {
    delete[] data;
    data = nullptr;
  }
}

template<typename T>
void DenseMatrix<T>::display(std::ostream& s) const {
  size_t const ld = this->ld;
  T const* const data = this->data;

  for ( size_t i = 0; i < this->m; i++ ) {
    for ( size_t j = 0; j < this->n; j++ ) {
      s << std::to_string(data[i + j*ld]) << " ";
    }
    s << "\n";
  }
}
