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

template <size_t Dimension = 2> class LocalLagrangeAssembler {
public:
  using Point = bg::model::point<double, Dimension, bg::cs::cartesian>;
  using Value = std::pair<Point, unsigned>;

  LocalLagrangeAssembler(const arma::mat &centers,
                         const size_t num_local_centers)
      : centers_(centers), num_local_centers_(num_local_centers) {
    assembleTree(); // Build up R Tree of nearest neighbor points so we can find
                    // local indices
  }

  arma::mat findLocalCenters(const arma::uvec &local_indices) {
    const size_t num_local_centers = local_indices.size();

    ///@todo srowe: Make this 2 a template parameter on dimension size, or
    // derive it dynamically at runtime
    arma::mat local_centers(num_local_centers, Dimension);

    ///@todo srowe: Is this simply a submatrix view we can easily extract via
    /// armadillo?
    for (size_t i = 0; i < num_local_centers; ++i) {
      local_centers.row(i) = centers_.row(local_indices(i));
    }
    return local_centers;
  }
  unsigned int findLocalIndex(const arma::mat &local_centers,
                              unsigned int index) {

    const arma::rowvec center = centers_.row(index);

    // Implement naive algorithm here. Upgrade later.
    // Machine precision equality errors possible.
    unsigned int local_index = 0;
    const size_t num_vectors = local_centers.n_rows;

    for (size_t i = 0; i < num_vectors; ++i) {
      auto matching = arma::all(center == local_centers.row(i));

      if (matching) {
        local_index = i;
      }
    }

    ///@todo srowe: If we fail to find this index, we return 0, which is wrong
    return local_index;
  }

  LocalLagrange<Dimension>
  generateLocalLagrangeFunction(const unsigned int index) {

    auto local_indices = getNearestNeighbors(index);
    auto local_centers = findLocalCenters(local_indices);
    unsigned int local_index = findLocalIndex(local_centers, index);

    LocalLagrange<Dimension> llf(local_centers, local_indices, local_index);

    return llf;
  }

  unsigned int num_centers() const { return num_centers_; }
  double scale_factor() const { return scale_factor_; }
  double mesh_norm() const { return mesh_norm_; }
  double ball_radius() const { return ball_radius_; }

  void assembleTree() {
    const size_t num_centers = centers_.n_rows;

    std::vector<Value> points;
    points.reserve(num_centers);

    for (size_t iter = 0; iter < num_centers; ++iter) {
      Point mypoint(centers_(iter, 0), centers_(iter, 1));
      points.emplace_back(std::move(mypoint), iter);
    }
    rt_.insert(points.begin(), points.end());
  }

  arma::uvec getNearestNeighbors(const unsigned int index) {
    // querying.
    Point center(centers_(index, 0), centers_(index, 1));
    Value center_value(center, index);
    std::vector<Value> neighbors;
    rt_.query(bgi::nearest(center, num_local_centers_),
              std::back_inserter(neighbors));

    size_t num_neighbors = neighbors.size();
    arma::uvec indices(num_neighbors);

    for (size_t i = 0; i < num_neighbors; ++i) {
      indices(i) = std::get<1>(neighbors[i]);
    }
    return indices;
  }

  void setScale_factor(double scale_factor) {
    scale_factor_ = scale_factor;
    updateBallRadius();
  }
  void setMesh_norm(double mesh_norm) {
    mesh_norm_ = mesh_norm;
    updateBallRadius();
  }
  void setCenters(const arma::mat &centers) {
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
