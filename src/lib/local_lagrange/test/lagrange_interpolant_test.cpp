#include "../local_lagrange.h"
#include "../local_lagrange_assembler.h"
#include "../local_lagrange_interpolant.h"
#include "../../math_utils/math_tools.h"
#include <gtest/gtest.h>

using namespace local_lagrange;

TEST(LocalLagrangeInterpolantTests, TestSimpleInterpolant) {
    std::vector<double> one_dim_points = mathtools::linspace<double>(0.0,1.0, 10.0);
    auto points = mathtools::meshgrid(one_dim_points, one_dim_points);
    const auto& x_centers = points[0];
    const auto& y_centers = points[1];

    auto local_lagrange_ensemble = buildLocalLagrangeFunctions(x_centers, y_centers, 10);

    // Choose a function sampled on the same point set
    size_t num_points = x_centers.size();
    arma::vec sampled_function(num_points);

    std::cout << "Hello" << std::endl;
    for (size_t i = 0; i < num_points; ++i) {
         sampled_function(i) = std::sin(2*M_PI*x_centers[i]) * std::cos(2*M_PI*y_centers[i]);
    }

    // Now that we have the function sampled, let's test it out on the ensemble
    std::cout << "Building interpolant" << std::endl;
    LocalLagrangeInterpolant interpolant(local_lagrange_ensemble, sampled_function);
    std::cout << "Interpolant built" << std::endl;
    for (size_t i = 0; i < num_points; ++i) {
    	const double x = x_centers[i];
    	const double y = y_centers[i];
    	EXPECT_NEAR(sampled_function(i), interpolant(x,y), 1e-13);
    }

}