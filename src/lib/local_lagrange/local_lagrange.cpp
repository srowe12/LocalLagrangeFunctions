#include "local_lagrange.h"
#include <math.h>
#include <armadillo>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>

#include <boost/geometry/index/rtree.hpp>


namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace local_lagrange {

typedef bg::model::point<double, 2, bg::cs::cartesian> point;
typedef std::pair<point, unsigned> value;

void LocalLagrangeAssembler::assembleTree() {
  std::vector<value> points;
  for (size_t iter = 0; iter < centers_x_.size(); iter++) {
    point mypoint(centers_x_[iter], centers_y_[iter]);
    value myvalue(mypoint, iter);
    points.push_back(myvalue);
  }
  rt_.insert(points.begin(), points.end());
}

arma::mat LocalLagrange::assembleInterpolationMatrix(
    const std::vector<double>& local_centers_x, const std::vector<double>& local_centers_y) {
  // Initialize matrix to all zeros.
  arma::mat interp_matrix(local_centers_x.size() + 3,
                          local_centers_x.size() + 3, arma::fill::zeros);
  double distx = 0;
  double disty = 0;
  double dist = 0;
  size_t num_centers = local_centers_x.size();
  for (size_t row = 0; row < num_centers; row++) {
    for (size_t col = row + 1; col < num_centers; col++) {
      distx = local_centers_x[row] - local_centers_x[col];
      disty = local_centers_y[row] - local_centers_y[col];
      dist = distx * distx + disty * disty;
      interp_matrix(row, col) = interp_matrix(col, row) = .5 * dist * log(dist);
    }
  }

  for (size_t row = 0; row < num_centers; row++) {
    interp_matrix(num_centers, row) = interp_matrix(row, num_centers) = 1;
    interp_matrix(num_centers + 1, row) = interp_matrix(row, num_centers + 1) =
        local_centers_x[row];
    interp_matrix(num_centers + 2, row) = interp_matrix(row, num_centers + 2) =
        local_centers_y[row];
  }

  // Move semantics maybe?
  return interp_matrix;
}

void LocalLagrange::buildCoefficients(const std::vector<double>& local_centers_x,
                                      const std::vector<double>& local_centers_y,
                                      unsigned int local_index) {

  //Work needed here perhaps. Is this bad generating interp_matrix in this function?
  arma::mat interp_matrix =
      assembleInterpolationMatrix(local_centers_x, local_centers_y);
  arma::vec rhs(local_centers_x.size() + 3, arma::fill::zeros);
  rhs(local_index) = 1;
  coefficients_ = arma::solve(interp_matrix, rhs);
}

std::array<std::vector<double>, 2>
LocalLagrangeAssembler::findLocalCenters(const std::vector<unsigned int>& local_indices) {

  size_t num_local_centers = local_indices.size();
  std::vector<double> local_x(num_local_centers);
  std::vector<double> local_y(num_local_centers);
  for (size_t i = 0; i < num_local_centers; i++) {
    local_x[i] = centers_x_[local_indices[i]];
    local_y[i] = centers_y_[local_indices[i]];
  }
  std::array<std::vector<double>, 2> local_centers{ local_x, local_y};
  return local_centers;
}

LocalLagrange
LocalLagrangeAssembler::generateLocalLagrangeFunction(unsigned int index) {

  LocalLagrange llf(index);

  std::vector<unsigned int> local_indices = getNearestNeighbors(index);
  std::array<std::vector<double>, 2> local_centers = findLocalCenters(local_indices);
  unsigned int local_index = findLocalIndex(local_centers, index);

  llf.setIndices(local_indices);
  llf.buildCoefficients(local_centers[0], local_centers[1], local_index);

  return llf;
}

unsigned int LocalLagrangeAssembler::findLocalIndex(
    const std::array<std::vector<double>, 2>& local_centers, unsigned int index) {
  double center_x = centers_x_[index];
  double center_y = centers_y_[index];
  // Implement naive algorithm here. Upgrade later.
  // Machine precision equality errors possible.
  unsigned int local_index = 0;
  for (size_t i = 0; i < local_centers[0].size(); i++) {
    if (center_x == local_centers[0][i] && center_y == local_centers[1][i]) {
      local_index = i;
      break;
    }
  }

  return local_index;
}

std::vector<unsigned>
LocalLagrangeAssembler::getNearestNeighbors(unsigned int index) {
  // Wrap values into a single point, then value pair. Pass into rt for
  // querying.
  point center(centers_x_[index], centers_y_[index]);
  value center_value(center, index);
  std::vector<value> neighbors;
  rt_.query(bgi::nearest(center, num_local_centers_), std::back_inserter(neighbors));

  std::vector<unsigned> indices;
  for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
    indices.push_back(std::get<1>(*it));
  }
  return indices;
}

void buildLocalLagrangeFunctions(const std::vector<double>& centers_x, const std::vector<double>& centers_y) {

}
} // namespace local_lagrange
