#ifndef LOCAL_LAGRANGE_INTERPOLATE_H
#define LOCAL_LAGRANGE_INTERPOLATE_H

#include <armadillo>

#include <math_utils/math_tools.h>

namespace rbf {

template <typename Kernel, size_t Dimension>
class RadialBasisFunctionInterpolant {
public:
  RadialBasisFunctionInterpolant(const Kernel& kernel,
                                 const arma::mat& centers,
                                 const arma::vec& sampled_data)
      : kernel_(kernel), centers_(centers) {
    buildCoefficients(sampled_data);
  }

  arma::vec interpolate(const arma::mat& data_points) {
    // Interpolation is given by sum_{i=1}^N c_i Phi(data, x)
    // These kernels usually are related by Phi(data, x) = phi(||data - x||)

    // Phi(data,x) = P, where P_ij = dist(data_i - center_j). Then P_ij * c_j
    // evaluates to the solution
    const auto num_data_points = data_points.n_rows;
    const auto num_centers = centers_.n_rows;

    arma::mat distance_matrix(num_data_points, num_centers);

    // Naive implemtation for now
    ///@todo srowe: Dimensionality mismatch???
    for (size_t i = 0; i < num_data_points; ++i) {
      for (size_t j = 0; j < num_centers; ++j) {
        distance_matrix(i, j) =
            kernel_(mathtools::computePointDistance<Dimension>(
                i, j, data_points, centers_));
      }
    }

    ///@todo srowe; If we're doing the sums above, we might as well just do the
    /// matrix multiplication as well
    return distance_matrix * coefficients_;
  }

private:
  void buildCoefficients(const arma::vec& sampled_data) {
    // Build Interpolation Matrix
    const size_t num_centers = centers_.n_rows;

    ///@todo: Only handle positive definite kernels for the moment
    arma::mat interpolation_matrix(num_centers, num_centers);
    // Compute distance matirx
    // Naive implementation now
    for (size_t row = 0; row < num_centers; ++row) {
      for (size_t col = row + 1; col < num_centers; ++col) {
        interpolation_matrix(row, col) =
            kernel_(mathtools::computeDistance<Dimension>(row, col, centers_));
        interpolation_matrix(col, row) = interpolation_matrix(row, col);
      }
    }

    const double diagonal_value = kernel_(0.0);

    interpolation_matrix.diag().fill(diagonal_value);

    // Solve linear system
    coefficients_ = arma::solve(interpolation_matrix, sampled_data);
  }

  Kernel kernel_;
  arma::mat centers_;
  arma::vec coefficients_;
};
}  // end namespace rbf

#endif  // LOCAL_LAGRANGE_INTERPOLATE_H
