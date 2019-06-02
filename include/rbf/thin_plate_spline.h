#ifndef LOCAL_LAGRANGE_THIN_PLATE_SPLINE_H
#define LOCAL_LAGRANGE_THIN_PLATE_SPLINE_H

#include <armadillo>
#include <omp.h>

template <size_t Dimenion = 2, size_t Degree = 1>
struct ThinPlateSpline {
  // r represents distance between points squared
#pragma omp declare simd
  inline double operator()(const double dist_squared) const {
    return .5 * dist_squared * std::log(dist_squared);
  }
};

#endif
