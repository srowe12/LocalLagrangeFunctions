#include <local_lagrange/local_lagrange.h>
#include <local_lagrange/local_lagrange_assembler.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <gtest/gtest.h>
#include <stdio.h>
#include <utility>

#include <math_utils/math_tools.h>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<double, 2, bg::cs::cartesian> point;
typedef std::pair<point, unsigned> value;

using namespace local_lagrange;

TEST(MyTest, AssembleInterpolationMatrix) {

  arma::mat centers{{1, 0}, {2, 1}, {3, 2}};

  local_lagrange::LocalLagrange<2> llf(0); // Index 0.
  arma::mat interp_matrix = llf.assembleInterpolationMatrix(centers);
  for (size_t i = 0; i < 6; i++) {
    EXPECT_EQ(0.0, interp_matrix(i, i));
  }
  EXPECT_EQ(3, arma::accu(interp_matrix.col(3)));
  EXPECT_EQ(6, arma::accu(interp_matrix.col(4)));
  EXPECT_EQ(3, arma::accu(interp_matrix.col(5)));
  EXPECT_EQ(3, arma::accu(interp_matrix.row(3)));
  EXPECT_EQ(6, arma::accu(interp_matrix.row(4)));
  EXPECT_EQ(3, arma::accu(interp_matrix.row(5)));
}

TEST(MyTest, SolveForCoefficients) {

  arma::mat centers{{1, 0}, {2, 1}, {3, 2}};
  unsigned int local_index = 0;
  local_lagrange::LocalLagrange<2> llf(0); // Index 0.
  llf.buildCoefficients(centers, 0);
  arma::vec coefs = llf.coefficients();

  arma::mat interp_matrix = llf.assembleInterpolationMatrix(centers);
  arma::vec rhs = interp_matrix * coefs;
  EXPECT_NEAR(1, rhs(local_index), 1e-13);
  rhs(local_index) = 0;
  for (auto it = rhs.begin(); it != rhs.end(); ++it) {
    EXPECT_NEAR(0, *it, 1e-13);
  }
}

TEST(MyTest, FindLocalIndexTest) {

  arma::mat centers{{1, 0}, {2, 1}, {3, 2}};

  local_lagrange::LocalLagrangeAssembler<2> llc(centers, 1);

  unsigned int index = 2;

  arma::mat local_centers{{2, 1}, {3, 2}};

  unsigned int local_index = llc.findLocalIndex(local_centers, index);
  EXPECT_EQ(1, local_index);
}

TEST(MyTest, BuildLocalLagrangeFunction) {

  size_t num_points = 50;

  auto xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);

  local_lagrange::LocalLagrangeAssembler<2> llc(centers, 1e-1);

  unsigned int index = 5;
  local_lagrange::LocalLagrange<2> llf =
      llc.generateLocalLagrangeFunction(index);
  arma::vec coefs = llf.coefficients();
  size_t num_coefs = coefs.n_rows - 3;
  EXPECT_NEAR(0, accu(coefs.subvec(0, num_coefs - 1)), 1e-10);
  double x_eval = 0;
  double y_eval = 0;
  arma::vec coef_tps = coefs.subvec(0, num_coefs - 1);
  const arma::mat local_centers = llf.centers();
  for (size_t iter = 0; iter < num_coefs; iter++) {
    x_eval += coef_tps(iter) * local_centers(iter, 0);
    y_eval += coef_tps(iter) * local_centers(iter, 1);
  }
  EXPECT_NEAR(0, x_eval, 1e-12);
  EXPECT_NEAR(0, y_eval, 1e-12);
}

TEST(LocalLagrangeTests, EvaluateOperator) {
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
    EXPECT_NEAR(expected_evaluations(i), llf(local_centers.row(i)), 1e-13);
  }
}

TEST(MyTest, BuildAllLocalLagrangeFunctions) {}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int return_value = RUN_ALL_TESTS();

  return return_value;
}
