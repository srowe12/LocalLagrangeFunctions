#ifndef LOCAL_LAGRANGE_HDR
#define LOCAL_LAGRANGE_HDR
#include <armadillo>
#include <array>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>

#include <boost/geometry/index/rtree.hpp>

// Namespace aliases for Boost
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace local_lagrange {

typedef bg::model::point<double, 2, bg::cs::cartesian> point;
typedef std::pair<point, unsigned> value;

class LocalLagrange {
public:
  LocalLagrange(
      const std::tuple<std::vector<double>, std::vector<double>> &local_centers,
      const std::vector<unsigned int> &local_indices,
      const unsigned int local_index)
      : index_(local_index), indices_(local_indices),
        centers_x_(std::get<0>(local_centers)),
        centers_y_(std::get<1>(local_centers)) {
    buildCoefficients(centers_x_, centers_y_, index_);
  }

  explicit LocalLagrange(unsigned int index) : index_(index) {}
  LocalLagrange(unsigned int index, std::vector<unsigned int> indices,
                std::vector<double> &coefs)
      : index_(index), indices_(indices), coefficients_(coefs) {}

  arma::mat assembleInterpolationMatrix(const arma::vec &local_centers_x,
                                        const arma::vec &local_centers_y);

  void buildCoefficients(const arma::vec &local_centers_x,
                         const arma::vec &local_centers_y,
                         unsigned int local_index);

  unsigned int index() const { return index_; }
  std::vector<unsigned int> indices() const { return indices_; }
  arma::vec coefficients() const { return coefficients_; }

  void setIndices(std::vector<unsigned int> indices) { indices_ = indices; }

  // Evaluates the Local Lagrange Function at a collection of points
  // The LLF evaluates via \sum_{i=1}^N c_i \|p -x_i\|^2 log(\|p - x_i\|)
  // where c_i are the coefficients in the vector coefficients_ and
  // x_i are the local_centers associated with the vector
  double operator()(const double x, const double y) {
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
        ///a conversion factor?
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

private:
  unsigned int index_;
  std::vector<unsigned int> indices_;
  arma::vec centers_x_;
  arma::vec centers_y_;
  arma::vec coefficients_;
};

class LocalLagrangeAssembler {
public:
  LocalLagrangeAssembler(const std::vector<double> &centers_x,
                         const std::vector<double> &centers_y,
                         const size_t num_local_centers)
      : centers_x_(centers_x), centers_y_(centers_y),
        num_local_centers_(num_local_centers) {
    assembleTree(); // Build up R Tree of nearest neighbor points so we can find
                    // local indices
  }

  std::tuple<std::vector<double>, std::vector<double>>
  findLocalCenters(const std::vector<unsigned int> &local_indices);
  unsigned int findLocalIndex(
      const std::tuple<std::vector<double>, std::vector<double>> &local_centers,
      unsigned int index);
  LocalLagrange generateLocalLagrangeFunction(const unsigned int index);

  unsigned int num_centers() const { return num_centers_; }
  double scale_factor() const { return scale_factor_; }
  double mesh_norm() const { return mesh_norm_; }
  double ball_radius() const { return ball_radius_; }

  std::vector<double> centers_x() const { return centers_x_; }
  std::vector<double> centers_y() const { return centers_y_; }

  void assembleTree();

  std::vector<unsigned> getNearestNeighbors(const unsigned int index);

  void setScale_factor(double scale_factor) {
    scale_factor_ = scale_factor;
    updateBallRadius();
  }
  void setMesh_norm(double mesh_norm) {
    mesh_norm_ = mesh_norm;
    updateBallRadius();
  }
  void setCenters(const std::vector<double> &centers_x,
                  const std::vector<double> &centers_y) {
    // Assumes size of centers_x and centers_y are the same
    centers_x_ = centers_x;
    centers_y_ = centers_y;
    num_centers_ = centers_x_.size();
  }

  void setNum_local_centers(const unsigned int num_local_centers) {
    num_local_centers_ = num_local_centers;
  }

private:
  void updateBallRadius() {
    ball_radius_ = scale_factor_ * mesh_norm_ * abs(log(mesh_norm_));
  }

  unsigned int num_centers_;
  double scale_factor_; // We use ball_radius =
                        // scale_factor*mesh_norm*abs(log(mesh_norm));
  double mesh_norm_;
  double ball_radius_;
  bgi::rtree<value, bgi::quadratic<16>> rt_; // R-tree for indexing points.

  std::vector<double> centers_x_;
  std::vector<double> centers_y_; // Assumes 2D structure.

  unsigned int num_local_centers_; // How many local centers should we find?
};

} // namespace local_lagrange
#endif // LOCAL_LAGRANGE_HDR
