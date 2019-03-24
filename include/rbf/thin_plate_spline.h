#ifndef LOCAL_LAGRANGE_THIN_PLATE_SPLINE_H
#define LOCAL_LAGRANGE_THIN_PLATE_SPLINE_H

#include <armadillo>
#include <math_tools.h>

template <size_t Dimension =2, size_t Degree =1>
void buildPolynomialMatrix(arma::mat& interpolation_matrix, const arma::mat& points) const {
    const num_rows = interpolation_matrix.n_rows;
    const num_cols = interpolation_matrix.n_cols;
    const num_points = points.n_rows;

    const size_t num_polynomials = computePolynomialBasis(Degree);

    // Interpolation matrix for conditionally positive definite kernel is of the form [A P; P^T 0]. 
    // Our goal is to fill in the matrix P. This form depends on the degree and basis;
    // The form of the matrix will in ascending order from lowest degree constant polynomial
    // to highest degree

    if (num_points + num_polynomials != num_points) {
      throw std::runtime_error("The size of the interpolation matrix is not equal to the number of points plus number of polynomial terms");
    }

    interpolation_matrix(arma::span(0,num_points-1), num_points) = 1.0; 
    for (size_t degree = 0; degree < Degree; ++degree) {
        // Given dim, and deg, find all subsets (x_1,x_2...x_dim) sums to deg
        // For degree we have x^i*y^degree-i
        // xyz poly would be for degree 1: x y z
        // For degree 2 we would have x^2 + xy + xz + y^2 + yz + z^2;
        // For degree 3 we would have x^3 + x^2 y + x^2z + x
        // (3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (0,3,0), (0,2,1), (0,1,2), (0,)
        for ()      

    }


  }
template <size_t Dimenion = 2, size_t Degree = 1>
struct ThinPlateSpline {

  // r represents distance between points squared
  inline double operator()(const double dist_squared) const {
    return .5 * dist_squared * std::log(dist_squared);
  }

};

#endif
