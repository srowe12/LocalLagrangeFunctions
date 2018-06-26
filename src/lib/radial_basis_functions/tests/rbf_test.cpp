#include "../interpolate.h"
#include <gtest/gtest.h>

#include "../../math_utils/math_tools.h" ///@todo srowe; Yikes, fix these relative includes

using namespace rbf;

TEST(RbfTest, ConstructoGaussianInterpolant) {
  // We build up a Gaussian interpolant for 2D data
  const double a = 0.0;
  const double b = 1.0;
  auto x_mesh = mathtools::linspace(a, b, 20);
  auto data = mathtools::meshgrid(x_mesh, x_mesh);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int return_value = RUN_ALL_TESTS();

  return return_value;
}
