#include "../gaussian.h"
#include "../interpolate.h"
#include <gtest/gtest.h>

#include "../../math_utils/math_tools.h" ///@todo srowe; Yikes, fix these relative includes

using namespace rbf;

TEST(RbfTest, ConstructoGaussianInterpolant) {
  // We build up a Gaussian interpolant for 2D data
  const double a = 0.0;
  const double b = 1.0;
  const int n = 20;
  auto x_mesh = mathtools::linspace(a, b, 20);
  const arma::mat data = mathtools::meshgrid(x_mesh, x_mesh);
  const size_t N = data.n_rows;
  // Use this data to build up a Gaussian interpolant

  Gaussian<double> gaussian(1.0);

  arma::vec sampled_data(20 * 20);
  for (size_t i = 0; i < data.n_rows; ++i) {
    sampled_data(i) = std::sin(data(i, 0)) * std::cos(data(i, 1));
  }

  RadialBasisFunctionInterpolant<Gaussian<double>, 1> interpolant(gaussian, data,
                                                      sampled_data);

  // Compute on the same dataset and see how well we do! We should get zeros back
  const auto interpolated_data = interpolant.interpolate(data);

  ASSERT_EQ(N, interpolated_data.n_rows);

  for (size_t i = 0; i < N ; ++i) {
  	EXPECT_NEAR(0.0, interpolated_data(i), 1e-9);
  }



}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int return_value = RUN_ALL_TESTS();

  return return_value;
}
