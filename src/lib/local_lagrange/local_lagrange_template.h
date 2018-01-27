#ifndef LOCAL_LAGRANGE_TEMPLATE_HDR
#define LOCAL_LAGRANGE_TEMPLATE_HDR
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

template <size_t n> class LocalLagrange {
public:
  explicit LocalLagrange(unsigned int index) : index_(index) {}
  LocalLagrange(unsigned int index, std::array<unsigned int, n> indices,
                std::array<double, n> &coefs)
      : index_(index), indices_(indices), coefficients_(coefs) {}

  arma::mat
  assembleInterpolationMatrix(const std::array<double, n> &local_centers_x,
                              const std::array<double, n> &local_centers_y) {

    arma::mat::fixed<n + 3, n + 3> interp_matrix(arma::fill::zeros);
    double distx = 0;
    double disty = 0;
    double dist = 0;
    for (size_t row = 0; row < n; row++) {
      for (size_t col = row + 1; col < n; col++) {
        distx = local_centers_x[row] - local_centers_x[col];
        disty = local_centers_y[row] - local_centers_y[col];
        dist = distx * distx + disty * disty;
        interp_matrix(row, col) = interp_matrix(col, row) =
            .5 * dist * log(dist);
      }
    }

    for (size_t row = 0; row < n; row++) {
      interp_matrix(n, row) = interp_matrix(row, n) = 1;
      interp_matrix(n + 1, row) = interp_matrix(row, n + 1) =
          local_centers_x[row];
      interp_matrix(n + 2, row) = interp_matrix(row, n + 2) =
          local_centers_y[row];
    }

    // Move semantics maybe?
    return interp_matrix;
  }

  void buildCoefficients(const std::array<double, n> &local_centers_x,
                         const std::array<double, n> &local_centers_y,
                         unsigned int local_index) {

    arma::mat interp_matrix =
        assembleInterpolationMatrix(local_centers_x, local_centers_y);
    arma::vec rhs(local_centers_x.size() + 3, arma::fill::zeros);
    rhs(local_index) = 1;
    coefficients_ = arma::solve(interp_matrix, rhs);
  }

  unsigned int index() const { return index_; }
  std::array<unsigned int, n> indices() const { return indices_; }
  arma::vec::fixed<n + 3> coefficients() const { return coefficients_; }

  void setIndices(std::array<unsigned int, n> indices) { indices_ = indices; }

private:
  unsigned int index_;
  std::array<unsigned int, n> indices_;
  arma::vec::fixed<n + 3> coefficients_;
};

template <size_t n> class LocalLagrangeAssembler {
public:
  LocalLagrangeAssembler() : num_centers_(0), scale_factor_(1), mesh_norm_(0) {
    updateBallRadius();
  }

  std::array<std::array<double, n>, 2>
  findLocalCenters(const std::array<unsigned int, n> &local_indices) {

    std::array<double, n> local_x;
    std::array<double, n> local_y;
    for (size_t i = 0; i < n; i++) {
      local_x[i] = centers_x_[local_indices[i]];
      local_y[i] = centers_y_[local_indices[i]];
    }
    std::array<std::array<double, n>, 2> local_centers{{local_x, local_y}};
    return local_centers;
  }

  unsigned int
  findLocalIndex(const std::array<std::array<double, n>, 2> &local_centers,
                 unsigned int index) {

    double center_x = centers_x_[index];
    double center_y = centers_y_[index];
    // Implement naive algorithm here. Upgrade later.
    // Machine precision equality errors possible.
    // TODO: Replace this with an STL algorithm to "find"
    unsigned int local_index = 0;
    for (size_t i = 0; i < local_centers[0].size(); i++) {
      if (center_x == local_centers[0][i] && center_y == local_centers[1][i]) {
        local_index = i;
        break;
      }
    }

    return local_index;
  }

  std::array<unsigned int, n> getNearestNeighbors(unsigned int index) {
    // Wrap values into a single point, then value pair. Pass into rt for
    // querying.
    point center(centers_x_[index], centers_y_[index]);
    value center_value(center, index);
    std::vector<value> neighbors;
    rt_.query(bgi::nearest(center, n), std::back_inserter(neighbors));

    std::array<unsigned, n> indices;
    size_t counter = 0;
    for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
      indices[counter] = std::get<1>(*it);
      counter += 1;
    }
    return indices;
  }
  LocalLagrange<n> generateLocalLagrangeFunction(const unsigned int index) {
    LocalLagrange<n> llf(index);
    std::array<unsigned int, n> local_indices = getNearestNeighbors(index);
    std::array<std::array<double, n>, 2> local_centers =
        findLocalCenters(local_indices);
    unsigned int local_index = findLocalIndex(local_centers, index);

    llf.setIndices(local_indices);
    llf.buildCoefficients(local_centers[0], local_centers[1], local_index);

    return llf;
  }

  unsigned int num_centers() const { return num_centers_; }
  double scale_factor() const { return scale_factor_; }
  double mesh_norm() const { return mesh_norm_; }
  double ball_radius() const { return ball_radius_; }
  std::vector<double> centers_x() const { return centers_x_; }
  std::vector<double> centers_y() const { return centers_y_; }

  void assembleTree() {

    std::vector<value> points;
    for (size_t iter = 0; iter < centers_x_.size(); iter++) {
      point mypoint(centers_x_[iter], centers_y_[iter]);
      value myvalue(mypoint, iter);
      points.push_back(myvalue);
    }
    rt_.insert(points.begin(), points.end());
  }

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
  double scale_factor_; // We use ball_radius =
                        // scale_factor*mesh_norm*abs(log(mesh_norm));
  double mesh_norm_;
  double ball_radius_;
  bgi::rtree<value, bgi::quadratic<16>> rt_; // R-tree for indexing points.

  std::vector<double> centers_x_;
  std::vector<double> centers_y_; // Assumes 2D structure.
};
} // namespace local_lagrange
#endif // LOCAL_LAGRANGE_TEMPLATE_HDR
