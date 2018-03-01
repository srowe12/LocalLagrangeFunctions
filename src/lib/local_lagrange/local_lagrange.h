#ifndef LOCAL_LAGRANGE_HDR
#define LOCAL_LAGRANGE_HDR
#include <armadillo>
#include <array>
#include <cmath>
#include <vector>

namespace local_lagrange {

template <size_t Coordinate>
double computeDistance(double &dist, const size_t row, const size_t col,
                       const arma::mat &points) {
  dist += (points(row, Coordinate) - points(col, Coordinate)) *
          (points(row, Coordinate) - points(col, Coordinate));
  return dist + computeDistance<Coordinate - 1>(dist, row, col, points);
}

template <>
double computeDistance<0>(double &dist, const size_t row, const size_t col,
                          const arma::mat &points) {
  dist += (points(row, 0) - points(col, 0)) * (points(row, 0) - points(col, 0));
  return dist;
}

template <size_t Dimension = 2> class LocalLagrange {
public:
  LocalLagrange(const arma::mat &local_centers, const arma::uvec &local_indices,
                const unsigned int local_index)
      : index_(local_index), indices_(local_indices), centers_(local_centers) {
    buildCoefficients(centers_, index_);
  }

  explicit LocalLagrange(unsigned int index) : index_(index) {}

  LocalLagrange(unsigned int index, const arma::uvec &indices,
                const arma::vec &coefs)
      : index_(index), indices_(indices), coefficients_(coefs) {}

  arma::mat assembleInterpolationMatrix(const arma::mat &local_centers) {
    // Initialize matrix to all zeros.

    size_t n_rows = local_centers.n_rows;
    arma::mat interp_matrix(n_rows + 3, n_rows + 3, arma::fill::zeros);
    double dist = 0.0;
    size_t num_centers = local_centers.n_rows;
    for (size_t row = 0; row < num_centers; row++) {
      for (size_t col = row + 1; col < num_centers; col++) {
        double dist = 0.0;
        computeDistance<Dimension - 1>(dist, row, col, local_centers);

        interp_matrix(row, col) = interp_matrix(col, row) =
            .5 * dist * std::log(dist);
      }
    }

    for (size_t row = 0; row < num_centers; row++) {
      interp_matrix(num_centers, row) = interp_matrix(row, num_centers) = 1;
      interp_matrix(num_centers + 1, row) =
          interp_matrix(row, num_centers + 1) = local_centers(row, 0);
      interp_matrix(num_centers + 2, row) =
          interp_matrix(row, num_centers + 2) = local_centers(row, 1);
    }

    return interp_matrix;
  }

  void buildCoefficients(const arma::mat &local_centers,
                         unsigned int local_index) {
    // Work needed here perhaps. Is this bad generating interp_matrix in this
    // function?
    arma::mat interp_matrix = assembleInterpolationMatrix(local_centers);
    arma::vec rhs(local_centers.n_rows + 3, arma::fill::zeros);
    rhs(local_index) = 1;
    coefficients_ = arma::solve(interp_matrix, rhs);
  }

  unsigned int index() const { return index_; }
  arma::uvec indices() const { return indices_; }
  arma::vec coefficients() const { return coefficients_; }

  void setIndices(const arma::uvec &indices) { indices_ = indices; }

  // Evaluates the Local Lagrange Function at a collection of points
  // The LLF evaluates via \sum_{i=1}^N c_i \|p -x_i\|^2 log(\|p - x_i\|)
  // where c_i are the coefficients in the vector coefficients_ and
  // x_i are the local_centers associated with the vector
  double operator()(const arma::rowvec::fixed<Dimension> &point) const {
    // Compute distance from points to position

    ///@todo srowe Naive implementation for now, improve on in the future
    double result = 0.0;
    const size_t num_centers = centers_.n_rows;
    for (size_t i = 0; i < num_centers; ++i) {
      const arma::rowvec::fixed<Dimension> diff = centers_.row(i) - point;
      const double distance = arma::dot(diff, diff);

      // Safety check for distance = 0
      if (distance != 0.0) {

        //.5 r^2 log(r^2) = r^2 log(r), this way we don't need square root
        ///@todo srowe: Would it be faster to convert this to std::log1p and use
        /// a conversion factor?
        result += coefficients_(i) * distance * std::log(distance);
      }
    }
    result *= .5; // Multiply in the 1/2 for the 1/2 * r^2 log(r^2)
    // With distance vector, compute r^2 log(r)

    // Polynomial is last three coefficients_ vector points; These are of the
    // form 1 + x + y
    const auto n_rows = coefficients_.n_rows;

    ///@todo srowe: Those constant values are incorrect for dimension != 2
    double polynomial_term =
        coefficients_(n_rows - 3) +
        arma::dot(coefficients_.rows(n_rows - 2, n_rows - 1), point);

    return result + polynomial_term;
  }

  void scaleCoefficients(const double scale_factor) {
    coefficients_ *= scale_factor;
  }

private:
  unsigned int index_;
  arma::uvec indices_;
  arma::mat centers_;
  arma::vec coefficients_;
};

} // namespace local_lagrange
#endif // LOCAL_LAGRANGE_HDR
