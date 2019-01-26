#include <local_lagrange/local_lagrange.h>
#include <local_lagrange/local_lagrange_assembler.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <gtest/gtest.h>
#include <stdio.h>
#include <string>
#include <utility>

#include <boost/geometry/index/rtree.hpp>
#include <math_utils/math_tools.h>

TEST(IntegrationTest, BuildAnLLF) {

  size_t num_points = 50;

  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);

  local_lagrange::LocalLagrangeAssembler<2> llc(centers, 200, 2e0);

  size_t iter = 1341;
  std::cout << "The iteration is " << iter << std::endl;
  local_lagrange::LocalLagrange<2> llf =
      llc.generateLocalLagrangeFunction(iter);
  arma::vec coefs = llf.coefficients();
  size_t num_coefs = coefs.n_rows - 3 - 1;
  EXPECT_NEAR(0, accu(coefs.subvec(0, num_coefs)), 1e-11);
  double x_eval = 0;
  double y_eval = 0;
  arma::vec coef_tps = coefs.subvec(0, num_coefs);
  arma::mat local_centers = llf.centers();
  for (size_t eval_iter = 0; eval_iter < num_coefs; eval_iter++) {
    x_eval += coef_tps(eval_iter) * local_centers(eval_iter, 0);
    y_eval += coef_tps(eval_iter) * local_centers(eval_iter, 1);
  }
  EXPECT_LT(x_eval, 1e-8);
  EXPECT_LT(y_eval, 1e-8);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int return_value = RUN_ALL_TESTS();

  return return_value;
}
