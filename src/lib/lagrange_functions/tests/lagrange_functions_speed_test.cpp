#include <lagrange/lagrange.h>

#include <math_utils/math_tools.h>
#include <string>
#include <utility>

#include <algorithm>
#include <chrono>
#include <rbf/thin_plate_spline.h>
double timeResults(const size_t num_points) {
  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);

  size_t numcenters = centers.n_rows;

  std::cout << "The total number of points being used is " << numcenters
            << "\n";
  auto start = std::chrono::steady_clock::now();

  ThinPlateSpline<2> tps;
  auto llfs =
      local_lagrange::computeLagrangeFunctions<2>(centers, tps);

  auto end = std::chrono::steady_clock::now();
  std::cout << "Run complete!" << std::endl;

  auto diff = end - start;
  double time_diff = std::chrono::duration<double, std::milli>(diff).count();

  std::cout << time_diff << "milliseconds" << std::endl;

  return time_diff;
}

int main() {
  // Purpose of this app is to time how long it takes to generate a full set of
  // Lagrange Functions

  std::vector<size_t> num_points_list{50, 75, 100, 125};
  std::vector<double> results_list;
  for (auto num_points : num_points_list) {
    results_list.push_back(timeResults(num_points));
  }

  return 0;
}
