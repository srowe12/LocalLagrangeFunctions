#include <local_lagrange/local_lagrange_interpolant.h>

#include <math_utils/math_tools.h>
#include <stdio.h>
#include <string>
#include <utility>

#include <algorithm>
#include <chrono>
#include <future>

double timeResults(const size_t num_points) {
  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);

  size_t numcenters = centers.n_rows;
  // (np.log(h))**2 / 10
  double h = std::sqrt(2) / 2.0 * 1.0 / num_points;
  double scale = 1.0 / 600.0;
  double mesh_size = scale * std::log(h) * std::log(h);

  std::cout << "The total number of points being used is " << numcenters
            << "\n";
  std::cout << "The mesh size is " << mesh_size << "\n";
  auto start = std::chrono::steady_clock::now();

  auto llfs =
      local_lagrange::buildLocalLagrangeFunctions<2>(centers, mesh_size);

  auto end = std::chrono::steady_clock::now();
  std::cout << "Run complete!" << std::endl;

  auto diff = end - start;
  double time_diff = std::chrono::duration<double, std::milli>(diff).count();

  std::cout << time_diff << "milliseconds" << std::endl;

  auto llfset = llfs.localLagrangeFunctions();
  std::vector<size_t> num_centers;
  size_t min = 999999;
  size_t max = 0;
  size_t mean = 0;
  for (const auto& llf : llfset) {
    int num = llf.centers().n_rows;
    num_centers.push_back(num);
    mean += num;
  }
  mean /= num_centers.size();
  auto it = std::max_element(num_centers.begin(), num_centers.end());
  auto it2 = std::min_element(num_centers.begin(), num_centers.end());
  std::cout << "The num centers is " << *it << " and the min is " << *it2
            << " and the average num is " << mean << "\n";
  return time_diff;
}

int main() {
  // Purpose of this app is to time how long it takes to generate a full set of
  // LLF's.

  // Build some centers; say 2601 of them.
  std::vector<size_t> num_points_list{50, 75}; // 100, 125};
  std::vector<double> results_list;
  for (auto num_points : num_points_list) {
    results_list.push_back(timeResults(num_points));
  }

  return 0;
}
