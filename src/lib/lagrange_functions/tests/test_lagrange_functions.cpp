#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <lagrange/lagrange.h>
#include <rbf/thin_plate_spline.h>

#include <math_utils/math_tools.h>

using namespace local_lagrange;

TEST_CASE("ComputeAllLagrangeFunctions") {
  // Fundamentally, coefficients * centers = eye(n,n)
  // We will test this on a small collection of points
  const size_t num_points = 10;
  const std::vector<double> xmesh =
      mathtools::linspace<double>(0, 1, num_points);
  const arma::mat centers = mathtools::meshgrid<double>(xmesh, xmesh);
  const ThinPlateSpline tps;
  const arma::mat coefficients =
      computeLagrangeFunctions<2, ThinPlateSpline>(centers, tps);

  arma::mat expected = arma::eye(centers.n_rows, centers.n_rows);

  const arma::mat result = coefficients * centers;

  const double error = arma::abs(result - expected).max();

  REQUIRE(error <= 1e-12);
}
