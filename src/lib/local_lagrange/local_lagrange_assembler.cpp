#include "local_lagrange_assembler.h"
#include <armadillo>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <math.h>

#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace local_lagrange {

void LocalLagrangeAssembler::assembleTree() {
  std::vector<Value> points;
  for (size_t iter = 0; iter < centers_x_.size(); iter++) {
    Point mypoint(centers_x_[iter], centers_y_[iter]);
    Value myvalue(mypoint, iter);
    points.push_back(myvalue);
  }
  rt_.insert(points.begin(), points.end());
}

std::tuple<arma::vec, arma::vec>
LocalLagrangeAssembler::findLocalCenters(const arma::uvec &local_indices) {

  const size_t num_local_centers = local_indices.size();
  arma::vec local_x(num_local_centers);
  arma::vec local_y(num_local_centers);
  for (size_t i = 0; i < num_local_centers; i++) {
    local_x[i] = centers_x_[local_indices[i]];
    local_y[i] = centers_y_[local_indices[i]];
  }
  return std::make_tuple(local_x, local_y);
}

LocalLagrange
LocalLagrangeAssembler::generateLocalLagrangeFunction(unsigned int index) {

  auto local_indices = getNearestNeighbors(index);
  auto local_centers = findLocalCenters(local_indices);
  unsigned int local_index = findLocalIndex(local_centers, index);

  LocalLagrange llf(local_centers, local_indices, local_index);

  return llf;
}

unsigned int LocalLagrangeAssembler::findLocalIndex(
    const std::tuple<arma::vec, arma::vec> &local_centers, unsigned int index) {

  const double center_x = centers_x_[index];
  const double center_y = centers_y_[index];
  // Implement naive algorithm here. Upgrade later.
  // Machine precision equality errors possible.
  unsigned int local_index = 0;
  const auto &local_centers_x = std::get<0>(local_centers);
  const auto &local_centers_y = std::get<1>(local_centers);
  const size_t num_vectors = local_centers_x.size();
  for (size_t i = 0; i < num_vectors; ++i) {
    if (center_x == local_centers_x[i] && center_y == local_centers_y[i]) {
      local_index = i;
      break;
    }
  }

  return local_index;
}

arma::uvec
LocalLagrangeAssembler::getNearestNeighbors(const unsigned int index) {
  // Wrap values into a single point, then value pair. Pass into rt for
  // querying.
  Point center(centers_x_[index], centers_y_[index]);
  Value center_value(center, index);
  std::vector<Value> neighbors;
  rt_.query(bgi::nearest(center, num_local_centers_),
            std::back_inserter(neighbors));

  // std::vector<unsigned> indices;
  // for (const auto neighbor : neighbors) {
  //   indices.emplace_back(
  //       std::get<1>(neighbor)); // Grab the index of the neighbor
  // }

  size_t num_neighbors = neighbors.size();
  arma::uvec indices(num_neighbors);

  for (size_t i = 0; i < num_neighbors; ++i) {
    indices(i) = std::get<1>(neighbors[i]);
  }
  return indices;
}

} // namespace local_lagrange
