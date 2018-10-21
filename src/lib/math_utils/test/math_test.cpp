#include <gtest/gtest.h>
#include <math_utils/math_tools.h>
#include <stdio.h>

TEST(MathTest, TestComputeLength) {
  arma::rowvec::fixed<3> v{1, 2, 3};
  const auto length = mathtools::computeLengthSquared<3>(v);
  EXPECT_DOUBLE_EQ(14, length);
}

TEST(MathTest, TestComputeLengthOneDim) {
  arma::rowvec::fixed<1> v{-2};
  const auto length = mathtools::computeLengthSquared<1>(v);
  EXPECT_DOUBLE_EQ(4, length);
}

TEST(MathTest, TestComputeSquaredDistance) {
  arma::rowvec::fixed<3> v1{1, 2, 3};
  arma::rowvec::fixed<3> v2{-3, -1, -2};

  const auto squared_distance = mathtools::computeSquaredDistance<3>(v1, v2);
  EXPECT_DOUBLE_EQ(4 * 4 + 3 * 3 + 5 * 5, squared_distance);
}

TEST(MathTest, ComputeDistanceRowVecs) {

  arma::mat points{{0, 1, 2}, {3, 4, 5}};

  const arma::rowvec p1{0, 1, 2};
  const arma::rowvec p2{3, 4, 5};
  double dist = mathtools::computeDistance<2>(p1, p2);

  double expected_dist = 3 * 3 + 3 * 3 + 3 * 3;
  EXPECT_EQ(expected_dist, dist);
}

TEST(MathTest, ComputeDistance) {

  arma::mat points{{0, 1, 2}, {3, 4, 5}};

  size_t row = 0;
  size_t col = 1;
  double dist = mathtools::computeDistance<2>(row, col, points);

  double expected_dist = 3 * 3 + 3 * 3 + 3 * 3;
  EXPECT_EQ(expected_dist, dist);
}

TEST(MathTest, ComputePointDistance) {

  arma::mat points{{0, 1, 2}, {3, 4, 5}};
  arma::mat other_points{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};

  size_t row = 1;
  size_t col = 2;
  double dist =
      mathtools::computePointDistance<2>(row, col, points, other_points);

  double expected_dist = 3 * 3 + 3 * 3 + 3 * 3;
  EXPECT_EQ(expected_dist, dist);
}

TEST(MathTest, LinspaceTest) {
  double a = 0;
  double b = 1;
  unsigned int num_points = 50;
  std::vector<double> points = mathtools::linspace<double>(a, b, num_points);
  EXPECT_EQ(num_points + 1, points.size());
  EXPECT_DOUBLE_EQ(a, points[0]);
  EXPECT_DOUBLE_EQ(b, points[points.size() - 1]);

  a = 3;
  b = 4;
  num_points = 23;
  std::vector<double> points_odd =
      mathtools::linspace<double>(a, b, num_points);
  EXPECT_EQ(num_points + 1, points_odd.size());
  EXPECT_DOUBLE_EQ(a, points_odd[0]);
  EXPECT_DOUBLE_EQ(b, points_odd[num_points]);
}

TEST(MathTest, MeshgridTest) {
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
  EXPECT_EQ(num_vals, xvals.n_rows);
  EXPECT_EQ(num_vals, yvals.n_rows);

  EXPECT_DOUBLE_EQ(xvals[0], ax);
  EXPECT_DOUBLE_EQ(xvals[num_vals - 1], bx);
  EXPECT_DOUBLE_EQ(xvals[3], 3.25);
  EXPECT_DOUBLE_EQ(xvals[num_vals - 4], 3.75);
  EXPECT_DOUBLE_EQ(yvals[0], ay);
  EXPECT_DOUBLE_EQ(yvals[num_vals - 1], by);
  EXPECT_DOUBLE_EQ(yvals[3], 1);
  EXPECT_DOUBLE_EQ(yvals[num_vals - 4], 2);
}

TEST(MathTest, WriteVectorTest) {
  std::vector<double> double_vec{3, 4, 5};
  mathtools::write_vector<double>(double_vec, "./double_vec.dat");
  std::ifstream infile("./double_vec.dat");
  std::vector<double> read_double_vec;
  double read_value;
  while (infile >> read_value) {
    read_double_vec.push_back(read_value);
  }
  ASSERT_EQ(double_vec.size(), read_double_vec.size());
  for (size_t iter = 0; iter < double_vec.size(); iter++) {
    EXPECT_EQ(double_vec[iter], read_double_vec[iter]);
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
  ASSERT_EQ(string_vec.size(), read_string_vec.size());
  for (size_t iter = 0; iter < string_vec.size(); iter++) {
    EXPECT_EQ(string_vec[iter], read_string_vec[iter]);
  }
  for (auto i = read_string_vec.begin(); i != read_string_vec.end(); ++i) {
    std::cout << *i << " " << std::endl;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int return_value = RUN_ALL_TESTS();

  return return_value;
}
