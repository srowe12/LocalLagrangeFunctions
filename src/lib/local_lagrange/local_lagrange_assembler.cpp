#include "local_lagrange_assembler.h"
#include <armadillo>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <math.h>

#include <boost/geometry/index/rtree.hpp>

#include <iostream>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace local_lagrange {

void LocalLagrangeAssembler::assembleTree() {
  
   const size_t num_centers = centers_.n_rows;

  std::vector<Value> points;

  points.reserve(num_centers);

  for (size_t iter = 0; iter < num_centers; ++iter) {
    Point mypoint(centers_(iter, 0), centers_(iter, 1));
    Value myvalue(mypoint, iter);
    points.push_back(myvalue);
  }
  rt_.insert(points.begin(), points.end());
}

arma::mat
LocalLagrangeAssembler::findLocalCenters(const arma::uvec &local_indices) {

  const size_t num_local_centers = local_indices.size();

  ///@todo srowe: Make this 2 a template parameter on dimension size, or
  // derive it dynamically at runtime
  arma::mat local_centers(num_local_centers, 2);

  ///@todo srowe: Is this simply a submatrix view we can easily extract via
  /// armadillo?
  for (size_t i = 0; i < num_local_centers; ++i) {
    local_centers(i, 0) = centers_(local_indices(i), 0);
    local_centers(i, 1) = centers_(local_indices(i), 1);
  }
  return local_centers;
}

LocalLagrange
LocalLagrangeAssembler::generateLocalLagrangeFunction(unsigned int index) {

  auto local_indices = getNearestNeighbors(index);
  auto local_centers = findLocalCenters(local_indices);
  unsigned int local_index = findLocalIndex(local_centers, index);

  LocalLagrange llf(local_centers, local_indices, local_index);

  return llf;
}

unsigned int
LocalLagrangeAssembler::findLocalIndex(const arma::mat &local_centers,
                                       unsigned int index) {

  const double center_x = centers_(index, 0);
  const double center_y = centers_(index, 1);
  // Implement naive algorithm here. Upgrade later.
  // Machine precision equality errors possible.
  unsigned int local_index = 0;
  const size_t num_vectors = local_centers.n_rows;

  for (size_t i = 0; i < num_vectors; ++i) {

    // if ((std::fabs(diff_x) < tol) && (std::fabs(diff_y) < tol)) {
    if ((center_x == local_centers(i, 0)) &&
        (center_y == local_centers(i, 1))) {
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

} // namespace local_lagrange
