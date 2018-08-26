#ifndef LOCAL_LAGRANGE_THIN_PLATE_SPLINE_H
#define LOCAL_LAGRANGE_THIN_PLATE_SPLINE_H

#include <armadillo>

struct ThinPlateSpline {

    // r represents distance between points squared
    double operator()(const double dist_squared) {
        return .5*dist_squared*std::log(dist_squared);
    }

};

#endif

