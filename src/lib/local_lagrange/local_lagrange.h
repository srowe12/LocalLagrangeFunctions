#include <stdio.h>
#include <math.h>
#include <vector>
#include <armadillo>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>

#include <boost/geometry/index/rtree.hpp>

// Namespace aliases for Boost
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace local_lagrange {

typedef bg::model::point<double, 2, bg::cs::cartesian> point;
typedef std::pair<point, unsigned> value;

class LocalLagrange {
 public:
  explicit LocalLagrange(unsigned int index) : index_(index) {}
  LocalLagrange(unsigned int index, std::vector<double> coefs,
                std::vector<double> indices)
      : index_(index), indices_(indices), coefficients_(coefs) {}

  arma::mat assembleInterpolationMatrix(std::vector<double> local_centers_x,
                                        std::vector<double> local_centers_y);

  void buildCoefficients(std::vector<double> local_centers_x,
                         std::vector<double> local_centers_y,
                         unsigned int local_index);

  unsigned int index() const { return index_; }
  std::vector<double> indices() const { return indices_; }
  arma::vec coefficients() const { return coefficients_; }

 private:
  unsigned int index_;
  std::vector<double> indices_;
  arma::vec coefficients_;
};

class LocalLagrangeConstructor {
 public:
  LocalLagrangeConstructor()
      : num_centers_(0), scale_factor_(1), mesh_norm_(0) {
    updateBallRadius();
  }

  LocalLagrange generateLocalLagrangeFunction(unsigned int index);

  unsigned int num_centers() const { return num_centers_; }
  double scale_factor() const { return scale_factor_; }
  double mesh_norm() const { return mesh_norm_; }
  double ball_radius() const { return ball_radius_; }

  std::vector<double> centers_x() const { return centers_x_; }
  std::vector<double> centers_y() const { return centers_y_; }

  void assembleTree();

  std::vector<unsigned> getNearestNeighbors(unsigned int index);

  void setScale_factor(double scale_factor) {
    scale_factor_ = scale_factor;
    updateBallRadius();
  }
  void setMesh_norm(double mesh_norm) {
    mesh_norm_ = mesh_norm;
    updateBallRadius();
  }
  void setCenters(std::vector<double> centers_x,
                  std::vector<double> centers_y) {
    // Assumes size of centers_x and centers_y are the same
    centers_x_ = centers_x;
    centers_y_ = centers_y;
    num_centers_ = centers_x_.size();
  }

 private:
  void updateBallRadius() {
    ball_radius_ = scale_factor_ * mesh_norm_ * abs(log(mesh_norm_));
  }

  unsigned int num_centers_;
  double scale_factor_;  // We use ball_radius =
                         // scale_factor*mesh_norm*abs(log(mesh_norm));
  double mesh_norm_;
  double ball_radius_;
  bgi::rtree<value, bgi::quadratic<16> > rt_;  // R-tree for indexing points.

  std::vector<double> centers_x_;
  std::vector<double> centers_y_;  // Assumes 2D structure.
};

}  // namespace local_lagrange
