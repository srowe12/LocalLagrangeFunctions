#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <rbf/gaussian.h>
#include <rbf/interpolate.h>

#include <math_utils/math_tools.h> 

using namespace rbf;

TEST_CASE("GaussianTest") {

  Gaussian<double> gaussian(1.0);

  const double val = gaussian(4.0);

  const double expected_val = std::exp(-1.0 * 4.0);
  REQUIRE( val == Approx(expecte_val).margin( 1e-12));
}

TEST_CASE("ConstructoGaussianInterpolant") {
  // We build up a Gaussian interpolant for 2D data
  const double a = 0.0;
  const double b = 1.0;
  const int n = 4;
  auto x_mesh = mathtools::linspace(a, b, 4);
  const arma::mat data = mathtools::meshgrid(x_mesh, x_mesh);
  const size_t N = data.n_rows;
  // Use this data to build up a Gaussian interpolant

  Gaussian<double> gaussian(4.0);

  arma::vec sampled_data(N);
  for (size_t i = 0; i < N; ++i) {
    sampled_data(i) = std::sin(data(i, 0)) * std::cos(data(i, 1));
  }

  RadialBasisFunctionInterpolant<Gaussian<double>, 1> interpolant(
      gaussian, data, sampled_data);

  // Compute on the same dataset and see how well we do! We should get zeros
  // back

  const arma::vec interpolated_data = interpolant.interpolate(data);

  REQUIRE(N ==interpolated_data.n_rows);

  for (size_t i = 0; i < N; ++i) {
    REQUIRE( interpolated_data(i) == Approx(sampled_data(i)).margin(1e-7));
  }
}

TEST_CASE("ConstructoGaussianInterpolantBigger") {
  // We build up a Gaussian interpolant for 2D data
  const double a = 0.0;
  const double b = 1.0;
  const int n = 20;
  auto x_mesh = mathtools::linspace(a, b, n);
  const arma::mat data = mathtools::meshgrid(x_mesh, x_mesh);
  const size_t N = data.n_rows;
  // Use this data to build up a Gaussian interpolant

  Gaussian<double> gaussian(4.0);

  arma::vec sampled_data(N);
  for (size_t i = 0; i < N; ++i) {
    sampled_data(i) = std::sin(data(i, 0)) * std::cos(data(i, 1));
  }

  RadialBasisFunctionInterpolant<Gaussian<double>, 1> interpolant(gaussian, data,
                                                      sampled_data);

  // Compute on the same dataset and see how well we do! We should get zeros back

  const arma::vec interpolated_data = interpolant.interpolate(data);

  REQUIRE(N == interpolated_data.n_rows);

  for (size_t i = 0; i < N ; ++i) {
    REQUIRE( interpolated_data(i) == Approx(sampled_data(i)).margin( 1e-7));
  }
}

TEST_CASE("ConstructoGaussianInterpolantConstant") {
  // We build up a Gaussian interpolant for 2D data
  const double a = 0.0;
  const double b = 1.0;
  const int n = 4;
  auto x_mesh = mathtools::linspace(a, b, 4);
  const arma::mat data = mathtools::meshgrid(x_mesh, x_mesh);
  const size_t N = data.n_rows;
  // Use this data to build up a Gaussian interpolant

  Gaussian<double> gaussian(4.0);

  arma::vec sampled_data(N);
  for (size_t i = 0; i < N; ++i) {
    sampled_data(i) = 1.0;
  }

  RadialBasisFunctionInterpolant<Gaussian<double>, 1> interpolant(
      gaussian, data, sampled_data);

  // Compute on the same dataset and see how well we do! We should get zeros
  // back

  const arma::vec interpolated_data = interpolant.interpolate(data);

  REQUIRE(N == interpolated_data.n_rows);

  for (size_t i = 0; i < N; ++i) {
    REQUIRE(interpolated_data(i) == Approx(sampled_data(i)).margin(1e-7));
  }
}

