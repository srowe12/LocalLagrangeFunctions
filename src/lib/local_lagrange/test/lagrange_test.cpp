#include <local_lagrange/local_lagrange.h>
#include <local_lagrange/local_lagrange_assembler.h>
#include <catch2/catch.hpp>

#include <stdio.h>
#include <utility>

#include <math_utils/math_tools.h>

using namespace local_lagrange;

double distance(const arma::rowvec& p1, const arma::rowvec& p2) {
  double dist = 0.0;
  for (size_t i = 0; i < p1.n_elem; ++i) {
    dist += (p1(i) - p2(i)) * (p1(i) - p2(i));
  }
  return dist;
}

TEST_CASE("AssembleInterpolationMatrix") {
  arma::mat centers{{1, 0}, {2, 1}, {3, 2}};

  local_lagrange::LocalLagrange<2> llf(0);  // Index 0.
  ThinPlateSpline<2> tps;
  arma::mat interp_matrix =
      computeInterpolationMatrix<2, ThinPlateSpline<2>>(centers, tps);
  for (size_t i = 0; i < 6; i++) {
    REQUIRE(0.0 == interp_matrix(i, i));
  }
  const arma::rowvec p0 = centers.row(0);
  const arma::rowvec p1 = centers.row(1);
  const arma::rowvec p2 = centers.row(2);
  double error01 = (interp_matrix(0, 1) - tps(distance(p0, p1)));
  double error02 = (interp_matrix(0, 2) - tps(distance(p0, p2)));
  double error12 = (interp_matrix(1, 2) - tps(distance(p1, p2)));

  REQUIRE(error01 <= 1e-8);
  REQUIRE(error02 <= 1e-8);
  REQUIRE(error12 <= 1e-8);

  interp_matrix.print("Interp matrix");

  const arma::vec x = centers.col(0);
  const arma::vec y = centers.col(1);

  const arma::mat upper_right =
      interp_matrix(arma::span(0, 2), arma::span(3, 5));
  arma::mat expected_upper_right = arma::zeros(3, 3);
  expected_upper_right.col(0).fill(1.0);
  expected_upper_right.col(1) = y;
  expected_upper_right.col(2) = x;

  const double upper_right_error = (expected_upper_right - upper_right).max();
  REQUIRE(upper_right_error <= 1e-8);

  const arma::mat lower_left =
      interp_matrix(arma::span(3, 5), arma::span(0, 2));
  const double lower_left_error = (expected_upper_right.t() - lower_left).max();
  REQUIRE(lower_left_error <= 1e-8);
}

TEST_CASE("SolveForCoefficients") {
  arma::mat centers{{1, 0}, {2, 1}, {3, 2}};
  unsigned int local_index = 0;
  local_lagrange::LocalLagrange<2> llf(0);  // Index 0.
  llf.buildCoefficients(centers, 0);
  arma::vec coefs = llf.coefficients();

  ThinPlateSpline<2> tps;
  arma::mat interp_matrix =
      computeInterpolationMatrix<2, ThinPlateSpline<2>>(centers, tps);
  arma::vec rhs = interp_matrix * coefs;
  REQUIRE(rhs(local_index) == Approx(1.0).margin(1e-13));
  rhs(local_index) = 0;
  for (auto it = rhs.begin(); it != rhs.end(); ++it) {
    REQUIRE(*it == Approx(0.0).margin(1e-13));
  }
}

TEST_CASE("FindLocalIndexTest") {
  arma::mat centers{{1, 0}, {2, 1}, {3, 2}};

  local_lagrange::LocalLagrangeAssembler<2> llc(centers, 1);

  unsigned int index = 2;

  arma::mat local_centers{{2, 1}, {3, 2}};

  unsigned int local_index = llc.findLocalIndex(local_centers, index);
  REQUIRE(1 == local_index);
}

TEST_CASE("BuildLocalLagrangeFunction") {
  size_t num_points = 50;

  auto xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);

  local_lagrange::LocalLagrangeAssembler<2> llc(centers, 1e-1);

  unsigned int index = 5;
  local_lagrange::LocalLagrange<2> llf =
      llc.generateLocalLagrangeFunction(index);
  arma::vec coefs = llf.coefficients();
  size_t num_coefs = coefs.n_rows - 3;
  REQUIRE(accu(coefs.subvec(0, num_coefs - 1)) == Approx(0.0).margin(1e-10));
  double x_eval = 0;
  double y_eval = 0;
  arma::vec coef_tps = coefs.subvec(0, num_coefs - 1);
  const arma::mat local_centers = llf.centers();
  for (size_t iter = 0; iter < num_coefs; iter++) {
    x_eval += coef_tps(iter) * local_centers(iter, 0);
    y_eval += coef_tps(iter) * local_centers(iter, 1);
  }
  REQUIRE(x_eval == Approx(0.0).margin(1e-12));
  REQUIRE(y_eval == Approx(0.0).margin(1e-12));
}

TEST_CASE("EvaluateOperator") {
  // Build an LLF centered at 5,5 out of 10 points in x and y directions

  arma::mat local_centers{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5},
                          {6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}};
  LocalLagrangeAssembler<2> assembler(local_centers, 100);

  auto llf = assembler.generateLocalLagrangeFunction(5);
  arma::vec expected_evaluations{0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  // The LLF should evaluate to 0 on all the centers except for
  // on the point the LLF is centered on, where it should be 1.0
  for (size_t i = 0; i < 10; ++i) {
    local_centers.row(i).print("The compared local center is");
    std::cout << " The i is " << i << " and the eval is "
              << llf(local_centers.row(i)) << "\n";
    REQUIRE(llf(local_centers.row(i)) ==
            Approx(expected_evaluations(i)).margin(1e-13));
  }
}

TEST_CASE("BuildAllLocalLagrangeFunctions") {
  // TODO: Add test for all Lagrange functions
}
