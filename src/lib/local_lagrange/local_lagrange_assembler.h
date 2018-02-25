#ifndef LOCAL_LAGRANGE_ASSEMBLER_HDR
#define LOCAL_LAGRANGE_ASSEMBLER_HDR
#include <armadillo>
#include <array>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>

#include <boost/geometry/index/rtree.hpp>

#include "local_lagrange.h"

// Namespace aliases for Boost
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace local_lagrange {

class LocalLagrangeAssembler {
public:
  using Point = bg::model::point<double, 2, bg::cs::cartesian>;
  using Value = std::pair<Point, unsigned>;

  LocalLagrangeAssembler(const arma::mat& centers,
                         const size_t num_local_centers)
      : centers_(centers),
        num_local_centers_(num_local_centers) {
    assembleTree(); // Build up R Tree of nearest neighbor points so we can find
                    // local indices
  }

  arma::mat
  findLocalCenters(const arma::uvec &local_indices);
  unsigned int
  findLocalIndex(const arma::mat &local_centers,
                 unsigned int index);
  LocalLagrange generateLocalLagrangeFunction(const unsigned int index);

  unsigned int num_centers() const { return num_centers_; }
  double scale_factor() const { return scale_factor_; }
  double mesh_norm() const { return mesh_norm_; }
  double ball_radius() const { return ball_radius_; }

  void assembleTree();

  arma::uvec getNearestNeighbors(const unsigned int index);

  void setScale_factor(double scale_factor) {
    scale_factor_ = scale_factor;
    updateBallRadius();
  }
  void setMesh_norm(double mesh_norm) {
    mesh_norm_ = mesh_norm;
    updateBallRadius();
  }
  void setCenters(const arma::mat& centers) {
    // Assumes size of centers_x and centers_y are the same
    centers_ = centers;
    num_centers_ = centers_.n_rows;
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
  bgi::rtree<Value, bgi::quadratic<16>> rt_; // R-tree for indexing points.

  arma::mat centers_; // N x d, with N points and d dimensions.

  unsigned int num_local_centers_; // How many local centers should we find?
};

} // namespace local_lagrange
#endif // LOCAL_LAGRANGE_ASSEMBLER_HDR
