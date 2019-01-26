#include <local_lagrange/local_lagrange_interpolant.h>

#include <gtest/gtest.h>
#include <math_utils/math_tools.h>
#include <stdio.h>
#include <string>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <chrono>
#include <future>

int main() {
  // Purpose of this app is to time how long it takes to generate a full set of
  // LLF's.

  // Build some centers; say 2601 of them.
  std::cout << "Beginning run!" << std::endl;
  auto start = std::chrono::steady_clock::now();
  size_t num_points = 50;
  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);

  size_t num_centers = centers.n_rows;

  auto llfs =
      local_lagrange::buildLocalLagrangeFunctions<2>(centers, 200, 1e-1);

  std::cout << "Run complete!" << std::endl;
  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  std::cout << std::chrono::duration<double, std::milli>(diff).count()
            << "milliseconds" << std::endl;

  return 0;
}
