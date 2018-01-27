#include "local_lagrange.h"
#include <armadillo>
#include <cmath>

namespace local_lagrange {

arma::mat
LocalLagrange::assembleInterpolationMatrix(const arma::vec &local_centers_x,
                                           const arma::vec &local_centers_y) {
  // Initialize matrix to all zeros.
  arma::mat interp_matrix(local_centers_x.size() + 3,
                          local_centers_x.size() + 3, arma::fill::zeros);
  double distx = 0;
  double disty = 0;
  double dist = 0;
  size_t num_centers = local_centers_x.size();
  for (size_t row = 0; row < num_centers; row++) {
    for (size_t col = row + 1; col < num_centers; col++) {
      distx = local_centers_x[row] - local_centers_x[col];
      disty = local_centers_y[row] - local_centers_y[col];
      dist = distx * distx + disty * disty;
      interp_matrix(row, col) = interp_matrix(col, row) =
          .5 * dist * std::log(dist);
    }
  }

  for (size_t row = 0; row < num_centers; row++) {
    interp_matrix(num_centers, row) = interp_matrix(row, num_centers) = 1;
    interp_matrix(num_centers + 1, row) = interp_matrix(row, num_centers + 1) =
        local_centers_x[row];
    interp_matrix(num_centers + 2, row) = interp_matrix(row, num_centers + 2) =
        local_centers_y[row];
  }

  // Move semantics maybe?
  return interp_matrix;
}

void LocalLagrange::buildCoefficients(const arma::vec &local_centers_x,
                                      const arma::vec &local_centers_y,
                                      unsigned int local_index) {

  // Work needed here perhaps. Is this bad generating interp_matrix in this
  // function?
  arma::mat interp_matrix =
      assembleInterpolationMatrix(local_centers_x, local_centers_y);
  arma::vec rhs(local_centers_x.size() + 3, arma::fill::zeros);
  rhs(local_index) = 1;
  coefficients_ = arma::solve(interp_matrix, rhs);
}

} // namespace local_lagrange
