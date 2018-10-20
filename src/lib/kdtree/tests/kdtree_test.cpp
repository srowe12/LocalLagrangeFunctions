#include <gtest/gtest.h>
#include <kdtree/kdtree.h>

#include <math_utils/math_tools.h>

TEST(KdtreeTests, SimpleTest) {

  arma::mat points{{0, 0}, {1, 1}, {0, 1}, {1, 0}, {.5, .5}};

  auto tree = BuildTree<2>(points);

  arma::mat point{0, 0};
  tree.search(tree, point);
}