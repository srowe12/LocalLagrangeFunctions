#ifndef LOCAL_LAGRANGE_HDR
#define LOCAL_LAGRANGE_HDR
#include <math_utils/math_tools.h>
#include <armadillo>
#include <array>
#include <cmath>
#include <vector>

#include <lagrange/lagrange.h>
#include <math_utils/polynomials.h>
#include <rbf/thin_plate_spline.h>

namespace local_lagrange {

template <size_t Dimension = 2, typename Kernel = ThinPlateSpline<Dimension>>
class LocalLagrange {
public:
  LocalLagrange(const arma::mat& local_centers, const unsigned int local_index)
      : index_(local_index), centers_(local_centers) {
    polynomial_powers_ = buildTuples<Dimension, 1>();
    buildCoefficients(centers_, index_);
  }

  LocalLagrange(const arma::mat& local_centers,
                const unsigned int local_index,
                const Tuples<Dimension>& polynomial_powers)
      : index_(local_index),
        centers_(local_centers),
        polynomial_powers_(polynomial_powers) {
    buildCoefficients(centers_, index_);
  }

  LocalLagrange() = default;

  void buildCoefficients(const arma::mat& local_centers,
                         unsigned int local_index) {
    const arma::mat interp_matrix =
        computeInterpolationMatrix<Dimension, Kernel>(local_centers, kernel_);
    arma::vec rhs(interp_matrix.n_cols, arma::fill::zeros);
    rhs(local_index) = 1;
    coefficients_ = arma::solve(interp_matrix, rhs, arma::solve_opts::fast);
  }

  unsigned int index() const { return index_; }
  arma::vec coefficients() const { return coefficients_; }
  arma::mat centers() const { return centers_; }

  // Evaluates the Local Lagrange Function at a collection of points
  // The LLF evaluates via \sum_{i=1}^N c_i \|p -x_i\|^2 log(\|p - x_i\|)
  // where c_i are the coefficients in the vector coefficients_ and
  // x_i are the local_centers associated with the vector
  double operator()(const arma::rowvec::fixed<Dimension>& point) const {
    // Compute distance from points to position

    ///@todo srowe Naive implementation for now, improve on in the future
    double result = 0.0;
    const size_t num_centers = centers_.n_rows;
    for (size_t i = 0; i < num_centers; ++i) {
      const double distance =
          mathtools::computeSquaredDistance<Dimension>(centers_.row(i), point);

      // Safety check for distance = 0
      if (distance != 0.0) {
        result += coefficients_(i) * kernel_(distance);
      }
    }

    // Polynomial is last three coefficients_ vector points; These are of the
    // form 1 + x + y
    const auto n_rows = coefficients_.n_rows;
    const auto num_polynomial_terms = coefficients_.n_rows - num_centers;
    double polynomial_term = polynomialApply<Dimension>(
        coefficients_.rows(n_rows - num_polynomial_terms, n_rows - num_polynomial_terms + 1), point, polynomial_powers_);

    ///@todo srowe: polynomial term is missing 0,0,0 power
    return result + coefficients_(n_rows - 1) + polynomial_term;
  }

  void scaleCoefficients(const double scale_factor) {
    coefficients_ *= scale_factor;
  }

private:
  unsigned int index_;
  Kernel kernel_;
  ///@todo srowe: Each LLF maintaing its centers is probably overkill
  arma::mat centers_;
  arma::vec coefficients_;
  Tuples<Dimension> polynomial_powers_;
};

}  // namespace local_lagrange
#endif  // LOCAL_LAGRANGE_HDR
