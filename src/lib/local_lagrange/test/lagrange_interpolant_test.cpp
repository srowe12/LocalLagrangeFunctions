#include "../../math_utils/math_tools.h"
#include "../local_lagrange.h"
#include "../local_lagrange_assembler.h"
#include "../local_lagrange_interpolant.h"
#include <gtest/gtest.h>

using namespace local_lagrange;

class LocalLagrangeInterpolantTests : public ::testing::Test {
protected:
  void SetUp() override {
    std::vector<double> one_dim_points =
        mathtools::linspace<double>(0.0, 1.0, 10.0);
    points = mathtools::meshgrid(one_dim_points, one_dim_points);


    // Choose a function sampled on the same point set
    num_points = x_centers.size();
    arma::vec sample_function(num_points);

    for (size_t i = 0; i < num_points; ++i) {
      sample_function(i) =
          std::sin(2 * M_PI * points[i,0]) * std::cos(2 * M_PI * points[i,1]);
    }

    sampled_function = sample_function;
  }

  size_t num_points;
  arma::mat points;

  arma::vec sampled_function;
};

TEST_F(LocalLagrangeInterpolantTests, TestSimpleInterpolant) {
  LocalLagrangeEnsemble local_lagrange_ensemble =
      buildLocalLagrangeFunctions(x_centers, y_centers, 500);

  // Now that we have the function sampled, let's test it out on the ensemble
  LocalLagrangeInterpolant interpolant(local_lagrange_ensemble,
                                       sampled_function);

  for (size_t i = 0; i < num_points; ++i) {
    const double x = centers[i,0];
    const double y = centers[i,1];

    EXPECT_NEAR(sampled_function(i), interpolant(x, y), 1e-13);
  }
}

TEST_F(LocalLagrangeInterpolantTests, OffgridPointEvaluation) {
  LocalLagrangeEnsemble local_lagrange_ensemble =
      buildLocalLagrangeFunctions(x_centers, y_centers, 500);

  // Now that we have the function sampled, let's test it out on the ensemble
  LocalLagrangeInterpolant interpolant(local_lagrange_ensemble,
                                       sampled_function);

  double some_x = 1.0 / 7.0;
  double some_y = 1.0 / 7.0;

  double value_at_point = interpolant(some_x, some_y);

  double expected_value = std::sin(2 * M_PI / 7.0) * std::cos(2 * M_PI / 7.0);

  // We only have 10 sample points in the x and y direction, so I don't expect
  // it to be incredibly accurate
  EXPECT_NEAR(expected_value, value_at_point, 2e-3);
}