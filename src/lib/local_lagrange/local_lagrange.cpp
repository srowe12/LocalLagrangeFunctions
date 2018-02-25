#include "local_lagrange.h"
#include <armadillo>
#include <cmath>

namespace local_lagrange {

arma::mat
LocalLagrange::assembleInterpolationMatrix(const arma::mat& local_centers) {
  // Initialize matrix to all zeros.

  size_t n_rows = local_centers.n_rows;
  arma::mat interp_matrix(n_rows + 3,
                          n_rows + 3, arma::fill::zeros);
  double distx = 0;
  double disty = 0;
  double dist = 0;
  size_t num_centers = local_centers.n_rows;
  for (size_t row = 0; row < num_centers; row++) {
    for (size_t col = row + 1; col < num_centers; col++) {
      distx = local_centers(row,0) - local_centers(col,0);
      disty = local_centers(row,1) - local_centers(col,1);
      dist = distx * distx + disty * disty;
      interp_matrix(row, col) = interp_matrix(col, row) =
          .5 * dist * std::log(dist);
    }
  }

  for (size_t row = 0; row < num_centers; row++) {
    interp_matrix(num_centers, row) = interp_matrix(row, num_centers) = 1;
    interp_matrix(num_centers + 1, row) = interp_matrix(row, num_centers + 1) =
        local_centers(row,0);
    interp_matrix(num_centers + 2, row) = interp_matrix(row, num_centers + 2) =
        local_centers(row,1);
  }

  return interp_matrix;
}

void LocalLagrange::buildCoefficients(const arma::mat& local_centers,
                                      unsigned int local_index) {

  // Work needed here perhaps. Is this bad generating interp_matrix in this
  // function?
  arma::mat interp_matrix =
      assembleInterpolationMatrix(local_centers);
  arma::vec rhs(local_centers.n_rows + 3, arma::fill::zeros);
  rhs(local_index) = 1;
  coefficients_ = arma::solve(interp_matrix, rhs);
}

} // namespace local_lagrange
