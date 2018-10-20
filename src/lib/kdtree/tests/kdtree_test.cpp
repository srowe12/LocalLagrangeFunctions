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
