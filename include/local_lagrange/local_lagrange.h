#ifndef LOCAL_LAGRANGE_HDR
#define LOCAL_LAGRANGE_HDR
#include <armadillo>
#include <array>
#include <cmath>
#include <math_utils/math_tools.h>
#include <vector>

#include <lagrange/lagrange.h>
#include <rbf/thin_plate_spline.h>

namespace local_lagrange {

template <size_t Dimension = 2, typename Kernel = ThinPlateSpline<Dimension>>
class LocalLagrange {
public:
  LocalLagrange(const arma::mat &local_centers, const unsigned int local_index)
      : index_(local_index), centers_(local_centers) {
    buildCoefficients(centers_, index_);
  }

  explicit LocalLagrange(unsigned int index) : index_(index) {}

  LocalLagrange(unsigned int index, const arma::vec &coefs)
      : index_(index), coefficients_(coefs) {}

  void buildCoefficients(const arma::mat &local_centers,
                         unsigned int local_index) {
    // Work needed here perhaps. Is this bad generating interp_matrix in this
    // function?
    Kernel kernel; ///@todo srowe: This is pretty derpy here
    const arma::mat interp_matrix =
        computeInterpolationMatrix<Dimension, Kernel>(local_centers, kernel);
    arma::vec rhs(local_centers.n_rows + 3, arma::fill::zeros);
    rhs(local_index) = 1;
    coefficients_ = arma::solve(interp_matrix, rhs);
  }

  unsigned int index() const { return index_; }
  arma::vec coefficients() const { return coefficients_; }
  arma::mat centers() const { return centers_; }

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

      const double distance =
          mathtools::computeSquaredDistance<Dimension>(centers_.row(i), point);

      // Safety check for distance = 0
      if (distance != 0.0) {

        ///@todo srowe: Make the Kernel a parameter

        //.5 r^2 log(r^2) = r^2 log(r), this way we don't need square root
        ///@todo srowe: Would it be faster to convert this to std::log1p and use
        /// a conversion factor?
        ///@todo srowe: REPLACE THIS WITH KERNEL!!!
        result += coefficients_(i) * distance * std::log(distance);
      }
    }
    result *= .5; // Multiply in the 1/2 for the 1/2 * r^2 log(r^2)
    // With distance vector, compute r^2 log(r)

    // Polynomial is last three coefficients_ vector points; These are of the
    // form 1 + x + y
    const auto n_rows = coefficients_.n_rows;

    ///@todo srowe: Make the polynomials more generic
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
  arma::mat centers_; ///@todo srowe: Each LLF maintaing its centers is probably
                      /// overkill
  arma::vec coefficients_;
};

} // namespace local_lagrange
#endif // LOCAL_LAGRANGE_HDR
