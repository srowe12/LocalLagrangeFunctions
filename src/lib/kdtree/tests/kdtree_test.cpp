#include <gtest/gtest.h>
#include <kdtree/kdtree.h>

#include <math_utils/math_tools.h>

TEST(KdtreeTests, SimpleTest) {

  arma::mat points{{0, 0}, {1, 1}, {0, 1}, {1, 0}, {.5, .5}};

  auto tree = BuildTree<2>(points);

  arma::rowvec point{0, 0};
  for (size_t i = 0; i < points.n_rows; ++i) {
    arma::rowvec point = points.row(i);
    bool in_tree = search<2>(tree, point);

    EXPECT_TRUE(in_tree);
  }

  arma::mat bad_point{{.3, .3}};
  bool in_tree = search<2>(tree, bad_point);
  EXPECT_FALSE(in_tree);
}

TEST(KdtreeTests, EvenNumberPointsBuildTree) {

  arma::mat points{{0, 0}, {1, 1}, {0, 1}, {1, 0}};

  auto tree = BuildTree<2>(points);
  arma::rowvec point{0, 0};
  for (size_t i = 0; i < points.n_rows; ++i) {
    arma::rowvec point = points.row(i);
    bool in_tree = search<2>(tree, point);

    EXPECT_TRUE(in_tree);
  }

  arma::mat bad_point{{.3, .3}};
  bool in_tree = search<2>(tree, bad_point);
  EXPECT_FALSE(in_tree);
}

TEST(KdtreeTests, RadiusQueryOnePoint) {

  arma::mat points{{0, 0}, {1, 1}, {0, 1}, {1, 0}, {.5, .5}};

  auto tree = BuildTree<2>(points);
  const double radius = .1;
  arma::rowvec point{.5, .5};

  const std::vector<arma::rowvec> found_points =
      RadiusQuery<2>(tree, point, radius);

  ASSERT_EQ(1, found_points.size());

  const double error = arma::norm(found_points[0] - point);

  EXPECT_NEAR(0.0, error, 1e-8);
}

bool inVector(const std::vector<arma::rowvec> &points,
              const arma::rowvec &point) {
  double min = 1000.0;
  for (const auto &p : points) {
    double diff = arma::norm(p - point);
    if (diff < min) {
      min = diff;
    }
  }

  return min < 1e-13;
}
TEST(KdtreeTests, RadiusQueryMultiplePoints) {
  arma::mat points{{0, 0}, {1, 1}, {0, 1}, {1, 0}, {.5, .5}};

  auto tree = BuildTree<2>(points);
  const double radius = .5;
  arma::rowvec point{.75, .75};

  const std::vector<arma::rowvec> found_points =
      RadiusQuery<2>(tree, point, radius);

  ASSERT_EQ(2, found_points.size());

  arma::rowvec expected_first_point{.5, .5};
  arma::rowvec expected_second_point{1, 1};
  arma::rowvec expected_third_point{.75, .7};

  EXPECT_TRUE(inVector(found_points, expected_first_point));
  EXPECT_TRUE(inVector(found_points, expected_second_point));
}