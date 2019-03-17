#ifndef LOCAL_LAGRANGE_THIN_PLATE_SPLINE_H
#define LOCAL_LAGRANGE_THIN_PLATE_SPLINE_H

#include <armadillo>

struct ThinPlateSpline {

  // r represents distance between points squared
  inline double operator()(const double dist_squared) const {
    return .5 * dist_squared * std::log(dist_squared);
  }
};

#endif
