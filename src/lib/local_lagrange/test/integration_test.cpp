#include "../local_lagrange.h"
#include "../local_lagrange_assembler.h"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <gtest/gtest.h>
#include <stdio.h>
#include <string>
#include <utility>

#include <boost/geometry/index/rtree.hpp>

#include "math_tools.h"

TEST(IntegrationTest, BuildAnLLF) {

  size_t num_points = 50;

  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);

  local_lagrange::LocalLagrangeAssembler<2> llc(centers, 200);

  // for (size_t iter = 0; iter < num_centers; iter++) {
  size_t iter = 1341;
  std::cout << "The iteration is " << iter << std::endl;
  local_lagrange::LocalLagrange llf = llc.generateLocalLagrangeFunction(iter);
  arma::vec coefs = llf.coefficients();
  EXPECT_NEAR(0, accu(coefs.subvec(0, 199)), 1e-11);
  double x_eval = 0;
  double y_eval = 0;
  arma::vec coef_tps = coefs.subvec(0, 199);
  auto local_indices = llf.indices();
  std::string index_file = "indices_" + std::to_string(iter) + ".txt";
  std::string coefs_file = "coefs_" + std::to_string(iter) + ".txt";
  // mathtools::write_vector(local_indices, index_file);
  // bool save_status = coefs.save(coefs_file, arma::raw_ascii);
  // EXPECT_TRUE(save_status);
  for (size_t eval_iter = 0; eval_iter < 200; eval_iter++) {
    x_eval += coef_tps(eval_iter) * centers[local_indices[eval_iter], 0];
    y_eval += coef_tps(eval_iter) * centers[local_indices[eval_iter], 1];
  }
  EXPECT_LT(x_eval, 1e-8);
  EXPECT_LT(y_eval, 1e-8);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int return_value = RUN_ALL_TESTS();

  return return_value;
}
