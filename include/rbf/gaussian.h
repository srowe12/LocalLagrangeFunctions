#ifndef LOCAL_LAGRANGE_GAUSSIAN_H
#define LOCAL_LAGRANGE_GAUSSIAN_H

#include <cmath>
#include <omp.h>

template <typename T>
struct Gaussian {
  Gaussian(const T val) : ScaleParameter(val) {}

#pragma omp declare simd
  inline T operator()(const T r_squared) const {
    return std::exp(-ScaleParameter * r_squared);
  }

  T ScaleParameter;
};
#endif  // LOCAL_LAGRANGE_GAUSSIAN_H
