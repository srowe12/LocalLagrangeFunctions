#include <gtest/gtest.h>
#include "../local_lagrange.h"
#include <stdio.h>
#include <utility>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>

#include <boost/geometry/index/rtree.hpp>

#include "math_tools.h"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<double, 2, bg::cs::cartesian> point;
typedef std::pair<point, unsigned> value;

using namespace local_lagrange;

TEST(MyTest, TreeTest) {
  std::vector<double> centers_x_{ 1, 2, 3 };
  std::vector<double> centers_y_{ 0, 1, 2 };

  bgi::rtree<value, bgi::quadratic<16> > rt;
  std::vector<value> points;
  for (size_t iter = 0; iter < centers_x_.size(); iter++) {
    point mypoint(centers_x_[iter], centers_y_[iter]);
    value myvalue(mypoint, iter);
    points.push_back(myvalue);
  }
  rt.insert(points.begin(), points.end());

  point origin(0, 0);
  std::vector<value> results;
  rt.query(bgi::nearest(origin, 2), std::back_inserter(results));
  EXPECT_EQ(2, results.size());
  std::pair<point, unsigned> first_value = results[0];
  point first_point = std::get<0>(first_value);
  EXPECT_EQ(2, first_point.get<0>());
  EXPECT_EQ(1, first_point.get<1>());

  point second_point = std::get<0>(results[1]);
  EXPECT_EQ(1, second_point.get<0>());
  EXPECT_EQ(0, second_point.get<1>());
}

TEST(MyTest, NearestNeighborTest) {

  std::vector<double> centers_x{ 1, 2, 3 };
  std::vector<double> centers_y{ 0, 1, 2 };
  LocalLagrangeAssembler llc(centers_x, centers_y, 2);

  std::vector<unsigned> indices = llc.getNearestNeighbors(0);
  EXPECT_EQ(2, indices.size());
  EXPECT_EQ(1, indices[0]);
  EXPECT_EQ(0, indices[1]);
}

TEST(MyTest, AssembleInterpolationMatrix) {
  std::vector<double> centers_x{ 1, 2, 3 };
  std::vector<double> centers_y{ 0, 1, 2 };
  local_lagrange::LocalLagrange llf(0); // Index 0.
  arma::mat interp_matrix =
      llf.assembleInterpolationMatrix(centers_x, centers_y);
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
  std::vector<double> centers_x{ 1, 2, 3 };
  std::vector<double> centers_y{ 0, 1, 2 };
  unsigned int local_index = 0;
  local_lagrange::LocalLagrange llf(0); // Index 0.
  llf.buildCoefficients(centers_x, centers_y, 0);
  arma::vec coefs = llf.coefficients();

  arma::mat interp_matrix =
      llf.assembleInterpolationMatrix(centers_x, centers_y);
  arma::vec rhs = interp_matrix * coefs;
  EXPECT_NEAR(1, rhs(local_index), 1e-13);
  rhs(local_index) = 0;
  for (auto it = rhs.begin(); it != rhs.end(); ++it) {
    EXPECT_NEAR(0, *it, 1e-13);
  }
}

TEST(MyTest, FindLocalIndexTest) {

  std::vector<double> centers_x{ 1, 2, 3 };
  std::vector<double> centers_y{ 0, 1, 2 };

  local_lagrange::LocalLagrangeAssembler llc(centers_x, centers_y, 2);

  unsigned int index = 2;
  std::vector<double> local_centers_x{2, 3};
  std::vector<double> local_centers_y{1, 2};
  auto local_centers = std::make_tuple(local_centers_x, local_centers_y);
  unsigned int local_index = llc.findLocalIndex(local_centers, index);
  EXPECT_EQ(1, local_index);
}

TEST(MyTest, FindLocalCentersTest) {
  size_t num_centers = 30;
  std::vector<double> centers_x(num_centers);
  std::vector<double> centers_y(num_centers);
  for (size_t i = 0; i < num_centers; i++) {
    centers_x[i] = i;
    centers_y[i] = i + 1;
  }
  local_lagrange::LocalLagrangeAssembler llc(centers_x, centers_y, 2);

  unsigned int index = 5;
  std::vector<unsigned int> local_indices = llc.getNearestNeighbors(index);
  auto local_centers = llc.findLocalCenters(local_indices);

  auto& local_centers_x = std::get<0>(local_centers);
  auto& local_centers_y = std::get<1>(local_centers);

  double center_x = centers_x[index];
  double center_y = centers_y[index];
  double dist;


  for (size_t i = 0; i < local_centers_x.size(); i++) {
    dist = (local_centers_x[i] - center_x) * (local_centers_x[i] - center_x) +
           (local_centers_y[i] - center_y) * (local_centers_y[i] - center_y);
    EXPECT_GT(2.0000001, dist);
  }
}

TEST(MyTest, BuildLocalLagrangeFunction) {

  size_t num_points = 50;

  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  std::array<std::vector<double>, 2> centers =
      mathtools::meshgrid<double>(xmesh, xmesh);

  local_lagrange::LocalLagrangeAssembler llc(centers[0], centers[1], 200);

  unsigned int index = 5;
  local_lagrange::LocalLagrange llf = llc.generateLocalLagrangeFunction(index);
  arma::vec coefs = llf.coefficients();
  EXPECT_NEAR(0, accu(coefs.subvec(0, 199)), 1e-10);
  double x_eval = 0;
  double y_eval = 0;
  arma::vec coef_tps = coefs.subvec(0, 199);
  std::vector<unsigned int> local_indices = llf.indices();

  for (size_t iter = 0; iter < 200; iter++) {
    x_eval += coef_tps(iter) * centers[0][local_indices[iter]];
    y_eval += coef_tps(iter) * centers[1][local_indices[iter]];
  }
  EXPECT_NEAR(0, x_eval, 1e-12);
  EXPECT_NEAR(0, y_eval, 1e-12);
}

TEST(MyTest, BuildAllLocalLagrangeFunctions) {

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int return_value = RUN_ALL_TESTS();

  return return_value;
}
