#include <lagrange/lagrange.h>
#include <rbf/thin_plate_spline.h>
#include <catch2/catch.hpp>

#include <math_utils/math_tools.h>

using namespace local_lagrange;

TEST_CASE("ComputeAllLagrangeFunctions") {
  // Fundamentally, coefficients * centers = eye(n,n)
  // We will test this on a small collection of points
  const size_t num_points = 10;
  const std::vector<double> xmesh =
      mathtools::linspace<double>(0, 1, num_points);
  const arma::mat centers = mathtools::meshgrid<double>(xmesh, xmesh);
  const ThinPlateSpline<2> tps;
  const arma::mat coefficients =
      computeLagrangeFunctions<2, ThinPlateSpline<2>>(centers, tps);

  // Coefficients in each col j satisfy
  // \sum_i=1^n coefs(i,j)*\Phi(\|x_k - x_i) + Poly(x_k)  = \delta_{k,j}

  // Build Up Interpolation Matrix

  const arma::mat interpolation_matrix =
      computeInterpolationMatrix<2, ThinPlateSpline<2>>(centers, tps);

  const arma::mat result = interpolation_matrix * coefficients;

  arma::mat expected = arma::eye(centers.n_rows + 3, centers.n_rows + 3);

  const size_t n = centers.n_rows;
  expected(n, n) = 0.0;
  expected(n + 1, n + 1) = 0.0;
  expected(n + 2, n + 2) = 0.0;

  double error = arma::abs(expected - result).max();

  REQUIRE(error <= 1e-12);
}
