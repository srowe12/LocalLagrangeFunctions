#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <local_lagrange/local_lagrange.h>
#include <local_lagrange/local_lagrange_assembler.h>

#include <stdio.h>
#include <string>
#include <utility>

#include <math_utils/math_tools.h>

TEST_CASE( "IntegrationTestBuildAnLLF") {

  size_t num_points = 50;

  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);

  local_lagrange::LocalLagrangeAssembler<2> llc(centers, 2e0);

  size_t iter = 1341;
  std::cout << "The iteration is " << iter << std::endl;
  local_lagrange::LocalLagrange<2> llf =
      llc.generateLocalLagrangeFunction(iter);
  arma::vec coefs = llf.coefficients();
  size_t num_coefs = coefs.n_rows - 3 - 1;
  REQUIRE(accu(coefs.subvec(0, num_coefs)) == Approx(0.0).margin(1e-10));
  double x_eval = 0;
  double y_eval = 0;
  arma::vec coef_tps = coefs.subvec(0, num_coefs);
  arma::mat local_centers = llf.centers();
  for (size_t eval_iter = 0; eval_iter < num_coefs; eval_iter++) {
    x_eval += coef_tps(eval_iter) * local_centers(eval_iter, 0);
    y_eval += coef_tps(eval_iter) * local_centers(eval_iter, 1);
  }
  REQUIRE(x_eval == Approx(0.0).margin(1e-8));
  REQUIRE(y_eval == Approx(0.0).margin(-8));
}
