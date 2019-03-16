#include <local_lagrange/local_lagrange_interpolant.h>

#include <math_utils/math_tools.h>
#include <stdio.h>
#include <string>
#include <utility>

#include <algorithm>
#include <chrono>
#include <future>

int main() {
  // Purpose of this app is to time how long it takes to generate a full set of
  // LLF's.

  // Build some centers; say 2601 of them.
  size_t num_points = 50;
  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);

  size_t numcenters = centers.n_rows;
  auto start = std::chrono::steady_clock::now();

  auto llfs = local_lagrange::buildLocalLagrangeFunctions<2>(centers, 3e-2);

  auto end = std::chrono::steady_clock::now();
  std::cout << "Run complete!" << std::endl;

  auto diff = end - start;
  std::cout << std::chrono::duration<double, std::milli>(diff).count()
            << "milliseconds" << std::endl;
  auto llfset = llfs.localLagrangeFunctions();
  std::vector<size_t> num_centers;
  size_t min = 999999;
  size_t max = 0;
  size_t mean = 0;
  for (const auto &llf : llfset) {
    int num = llf.centers().n_rows;
    num_centers.push_back(num);
    mean += num;
  }
  mean /= num_centers.size();
  auto it = std::max_element(num_centers.begin(), num_centers.end());
  auto it2 = std::min_element(num_centers.begin(), num_centers.end());
  std::cout << "The num centers is " << *it << " and the min is " << *it2
            << " and the average num is " << mean << "\n";
  return 0;
}
