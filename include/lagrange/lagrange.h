#ifndef LOCAL_LAGRANGE_LAGRANGE_H
#define LOCAL_LAGRANGE_LAGRANGE_H

#include <math_utils/math_tools.h>
#include <armadillo>
#include <array>
#include <cmath>
#include <vector>
#include <math_utils/polynomials.h>

namespace local_lagrange {

template <size_t Dimension, typename Kernel>
arma::mat computeInterpolationMatrix(const arma::mat& centers,
                                     const Kernel& kernel) {
  const size_t num_centers = centers.n_rows;

  ///@todo srowe: The 3 is not super right
  arma::mat interp_matrix(num_centers + 3, num_centers + 3, arma::fill::zeros);
  double dist = 0.0;
  for (size_t row = 0; row < num_centers; ++row) {
    for (size_t col = row + 1; col < num_centers; ++col) {
      dist = mathtools::computeDistance<Dimension - 1>(row, col, centers);

      interp_matrix(row, col) = interp_matrix(col, row) = kernel(dist);
    }
  }
  
  ///@todo srowe: Degree should be computable
  buildPolynomialMatrix<Dimension, 1>(interp_matrix, centers); 

  return interp_matrix;
}

template <size_t Dimension, typename Kernel>
arma::mat computeLagrangeFunctions(const arma::mat& centers,
                                   const Kernel& kernel) {
  // The Lagrange Functions are computed by forming a full distance matrix
  // applying the kernel to it, and solving Ax = b where b is a diagonal matrix.

  arma::mat interpolation_matrix =
      computeInterpolationMatrix<Dimension, Kernel>(centers, kernel);
  const int num_points = centers.n_rows;
  arma::mat rhs = arma::eye(num_points + 3, num_points + 3);
  // Zero out the diagonal for the polynomial terms.
  rhs(num_points, num_points) = 0;
  rhs(num_points + 1, num_points + 1) = 0;
  rhs(num_points + 2, num_points + 2) = 0;
  arma::mat coefficients = arma::solve(interpolation_matrix, rhs);

  return coefficients;
}
}  // namespace local_lagrange
#endif  // LOCAL_LAGRANGE_LAGRANGE_H
