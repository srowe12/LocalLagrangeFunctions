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
  bgi::rtree<Value, bgi::quadratic<16>> rt_; // R-tree for indexing points.

  std::vector<double> centers_x_;
  std::vector<double> centers_y_; // Assumes 2D structure.

  unsigned int num_local_centers_; // How many local centers should we find?
};

} // namespace local_lagrange
#endif // LOCAL_LAGRANGE_ASSEMBLER_HDR
