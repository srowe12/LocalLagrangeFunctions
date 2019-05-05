#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <local_lagrange/local_lagrange.h>
#include <local_lagrange/local_lagrange_assembler.h>
#include <local_lagrange/local_lagrange_interpolant.h>
#include <math_utils/math_tools.h>

using namespace local_lagrange;

std::pair<arma::mat, arma::vec> setupPoints() {
    std::vector<double> one_dim_points =
        mathtools::linspace<double>(0.0, 1.0, 10.0);
    arma::mat points = mathtools::meshgrid(one_dim_points, one_dim_points);

    // Choose a function sampled on the same point set
    auto num_points = points.n_rows;
    arma::vec sample_function(num_points);

    for (size_t i = 0; i < num_points; ++i) {
      sample_function(i) =
          std::sin(2 * M_PI * points(i, 0)) * std::cos(2 * M_PI * points(i, 1));
    }

    return std::make_pair(points, sample_function);
}

TEST_CASE("TestSimpleInterpolant") {
  auto [points, sampled_function] = setupPoints();
  LocalLagrangeEnsemble<2> local_lagrange_ensemble =
      buildLocalLagrangeFunctions<2>(points, 2e0);

  // Now that we have the function sampled, let's test it out on the ensemble
  LocalLagrangeInterpolant<2> interpolant(local_lagrange_ensemble,
                                          sampled_function);
  auto num_points = points.n_rows;
  for (size_t i = 0; i < num_points; ++i) {
    std::cout << "i = " << i << " interpolant() " << interpolant(points.row(i)) << " and sampled is " << sampled_function(i) << "\n";
    REQUIRE(interpolant(points.row(i)) == Approx(sampled_function(i)).margin( 1e-13));
  }
}

TEST_CASE("OffgridPointEvaluation") {
    auto [points, sampled_function] = setupPoints();

  LocalLagrangeEnsemble<2> local_lagrange_ensemble =
      buildLocalLagrangeFunctions<2>(points, 2e0);

  // Now that we have the function sampled, let's test it out on the ensemble
  LocalLagrangeInterpolant<2> interpolant(local_lagrange_ensemble,
                                          sampled_function);

  double some_x = 1.0 / 7.0;
  double some_y = 1.0 / 7.0;

  double value_at_point = interpolant(arma::rowvec{some_x, some_y});

  double expected_value = std::sin(2 * M_PI / 7.0) * std::cos(2 * M_PI / 7.0);

  // We only have 10 sample points in the x and y direction, so I don't expect
  // it to be incredibly accurate
  REQUIRE(value_at_point == Approx(expected_value).margin( 2e-3));
}
