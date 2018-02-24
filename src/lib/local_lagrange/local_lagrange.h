#ifndef LOCAL_LAGRANGE_HDR
#define LOCAL_LAGRANGE_HDR
#include <armadillo>
#include <array>
#include <cmath>
#include <vector>

namespace local_lagrange {

class LocalLagrange {
public:
  LocalLagrange(const std::tuple<arma::vec, arma::vec> &local_centers,
                const arma::uvec &local_indices, const unsigned int local_index)
      : index_(local_index), indices_(local_indices),
        centers_x_(std::get<0>(local_centers)),
        centers_y_(std::get<1>(local_centers)) {
    buildCoefficients(centers_x_, centers_y_, index_);
  }

  explicit LocalLagrange(unsigned int index) : index_(index) {}

  LocalLagrange(unsigned int index, const arma::uvec &indices,
                const std::vector<double> &coefs)
      : index_(index), indices_(indices), coefficients_(coefs) {}

  arma::mat assembleInterpolationMatrix(const arma::vec &local_centers_x,
                                        const arma::vec &local_centers_y);

  void buildCoefficients(const arma::vec &local_centers_x,
                         const arma::vec &local_centers_y,
                         unsigned int local_index);

  unsigned int index() const { return index_; }
  arma::uvec indices() const { return indices_; }
  arma::vec coefficients() const { return coefficients_; }

  void setIndices(const arma::uvec &indices) { indices_ = indices; }

  // Evaluates the Local Lagrange Function at a collection of points
  // The LLF evaluates via \sum_{i=1}^N c_i \|p -x_i\|^2 log(\|p - x_i\|)
  // where c_i are the coefficients in the vector coefficients_ and
  // x_i are the local_centers associated with the vector
  double operator()(const double x, const double y) const {
    // Compute distance from points to position

    ///@todo srowe Naive implementation for now, improve on in the future
    double result = 0.0;
    for (size_t i = 0; i < centers_x_.size(); ++i) {
      const double xdist = centers_x_[i] - x;
      const double ydist = centers_y_[i] - y;
      const double distance = xdist * xdist + ydist * ydist;

      // Safety check for distance = 0
      if (distance != 0.0) {

        //.5 r^2 log(r^2) = r^2 log(r), this way we don't need square root
        ///@todo srowe: Would it be faster to convert this to std::log1p and use
        /// a conversion factor?
        result += coefficients_[i] * distance * std::log(distance);
      }
    }
    result *= .5; // Multiply in the 1/2 for the 1/2 * r^2 log(r^2)
    // With distance vector, compute r^2 log(r)

    ///@todo srowe: We need to also compute in the Polynomial terms!

    // Polynomial is last three coefficients_ vector points; These are of the
    // form 1 + x + y
    const auto n_rows = coefficients_.n_rows;

    double polynomial_term = coefficients_[n_rows - 3] +
                             coefficients_[n_rows - 2] * x +
                             coefficients_[n_rows - 1] * y;

    return result + polynomial_term;
  }

  void scaleCoefficients(const double scale_factor) {
    coefficients_ *= scale_factor;
  }

private:
  unsigned int index_;
  arma::uvec indices_;
  arma::vec centers_x_;
  arma::vec centers_y_;
  arma::vec coefficients_;
};

} // namespace local_lagrange
#endif // LOCAL_LAGRANGE_HDR
