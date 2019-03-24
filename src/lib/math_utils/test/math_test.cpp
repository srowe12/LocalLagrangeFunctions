#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <math_utils/math_tools.h>
#include <stdio.h>

using namespace mathtools;
TEST_CASE("FactorialTestZero") {
  size_t n = 0;
  REQUIRE(1 == factorial(n));
}

TEST_CASE("FactorialTestOne") { REQUIRE(1 == factorial(1)); }

TEST_CASE("FactorialTestLarger") { REQUIRE(4 * 3 * 2 == factorial(4)); }

TEST_CASE("PolynomialSize") {
  size_t degree = 3;
  // basis is 1  ,x ,y, x2, xy, y2, x3 ,x2y, xy2, y3
  size_t size_poly_basis = computePolynomialBasis<2>(degree);
  REQUIRE(10 == size_poly_basis);
}

TEST_CASE("TestComputeLength") {
  arma::rowvec::fixed<3> v{1, 2, 3};
  const auto length = mathtools::computeLengthSquared<3>(v);
  REQUIRE(14 == length);
}

TEST_CASE("TestComputeLengthOneDim") {
  arma::rowvec::fixed<1> v{-2};
  const auto length = mathtools::computeLengthSquared<1>(v);
  REQUIRE(4 == length);
}

TEST_CASE("TestComputeSquaredDistance") {
  arma::rowvec::fixed<3> v1{1, 2, 3};
  arma::rowvec::fixed<3> v2{-3, -1, -2};

  const auto squared_distance = mathtools::computeSquaredDistance<3>(v1, v2);
  REQUIRE(4 * 4 + 3 * 3 + 5 * 5 == squared_distance);
}

TEST_CASE("ComputeDistanceRowVecs") {

  arma::mat points{{0, 1, 2}, {3, 4, 5}};

  const arma::rowvec p1{0, 1, 2};
  const arma::rowvec p2{3, 4, 5};
  double dist = mathtools::computeDistance<2>(p1, p2);

  double expected_dist = 3 * 3 + 3 * 3 + 3 * 3;
  REQUIRE(expected_dist == dist);
}

TEST_CASE("ComputeDistance") {

  arma::mat points{{0, 1, 2}, {3, 4, 5}};

  size_t row = 0;
  size_t col = 1;
  double dist = mathtools::computeDistance<2>(row, col, points);

  double expected_dist = 3 * 3 + 3 * 3 + 3 * 3;
  REQUIRE(expected_dist == dist);
}

TEST_CASE("ComputePointDistance") {

  arma::mat points{{0, 1, 2}, {3, 4, 5}};
  arma::mat other_points{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};

  size_t row = 1;
  size_t col = 2;
  double dist =
      mathtools::computePointDistance<2>(row, col, points, other_points);

  double expected_dist = 3 * 3 + 3 * 3 + 3 * 3;
  REQUIRE(expected_dist == dist);
}

TEST_CASE("LinspaceTest") {
  double a = 0;
  double b = 1;
  unsigned int num_points = 50;
  std::vector<double> points = mathtools::linspace<double>(a, b, num_points);
  REQUIRE(num_points + 1 == points.size());
  REQUIRE(a == points[0]);
  REQUIRE(b == points[points.size() - 1]);

  a = 3;
  b = 4;
  num_points = 23;
  std::vector<double> points_odd =
      mathtools::linspace<double>(a, b, num_points);
  REQUIRE(num_points + 1 == points_odd.size());
  REQUIRE(a == points_odd[0]);
  REQUIRE(b == points_odd[num_points]);
}

TEST_CASE("MeshgridTest") {
  double ax = 3;
  double bx = 4;
  size_t num_x = 4;
  std::vector<double> xpoints = mathtools::linspace<double>(ax, bx, num_x);

  double ay = 1;
  double by = 2;
  size_t num_y = 2;
  std::vector<double> ypoints = mathtools::linspace<double>(ay, by, num_y);

  auto pointset = mathtools::meshgrid<double>(xpoints, ypoints);
  auto xvals = pointset.col(0);
  auto yvals = pointset.col(1);

  size_t num_vals = (num_x + 1) * (num_y + 1);
  REQUIRE(num_vals == xvals.n_rows);
  REQUIRE(num_vals == yvals.n_rows);

  REQUIRE(xvals[0] == ax);
  REQUIRE(xvals[num_vals - 1] == bx);
  REQUIRE(xvals[3] == 3.25);
  REQUIRE(xvals[num_vals - 4] == 3.75);
  REQUIRE(yvals[0] == ay);
  REQUIRE(yvals[num_vals - 1] == by);
  REQUIRE(yvals[3] == 1);
  REQUIRE(yvals[num_vals - 4] == 2);
}

TEST_CASE("WriteVectorTest") {
  std::vector<double> double_vec{3, 4, 5};
  mathtools::write_vector<double>(double_vec, "./double_vec.dat");
  std::ifstream infile("./double_vec.dat");
  std::vector<double> read_double_vec;
  double read_value;
  while (infile >> read_value) {
    read_double_vec.push_back(read_value);
  }
  REQUIRE(double_vec.size() == read_double_vec.size());
  for (size_t iter = 0; iter < double_vec.size(); iter++) {
    REQUIRE(double_vec[iter] == read_double_vec[iter]);
  }
  for (auto i = read_double_vec.begin(); i != read_double_vec.end(); ++i) {
    std::cout << *i << " " << std::endl;
  }

  std::vector<std::string> string_vec{"The lol", "Line 2", "This is line 3"};

  mathtools::write_vector<std::string>(string_vec, "./string_vec.dat");
  std::ifstream infile_string("./string_vec.dat");
  std::vector<std::string> read_string_vec;
  std::string read_string;
  while (std::getline(infile_string, read_string)) {
    read_string_vec.push_back(read_string);
  }
  REQUIRE(string_vec.size() == read_string_vec.size());
  for (size_t iter = 0; iter < string_vec.size(); iter++) {
    REQUIRE(string_vec[iter] == read_string_vec[iter]);
  }
  for (auto i = read_string_vec.begin(); i != read_string_vec.end(); ++i) {
    std::cout << *i << " " << std::endl;
  }
}

TEST_CASE("FindTuplesDim3Degree1") {
   // Expect (1,0,0), (0,1,0), (0,0,1)
   std::vector<std::array<size_t, 3>> results;
   findtuples<3>(results, 1,0);

   std::cout << "The size of results is " << results.size() << "\n";
   for (auto& v: results) {
     std::cout << v[0] << " , " << v[1] << " , " << v[2] << "\n";
   }
}

TEST_CASE("FindtuplesDim3Degree2") {
  // Expect (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,2,0)


}