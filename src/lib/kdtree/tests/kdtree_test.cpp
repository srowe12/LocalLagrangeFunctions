#include <gtest/gtest.h>
#include <kdtree/kdtree.h>

#include <math_utils/math_tools.h>

TEST(KdtreeTests, SimpleTest) {

  arma::mat points{{0, 0}, {1, 1}, {0, 1}, {1, 0}, {.5, .5}};

  std::cout << "Starting tree" << std::endl;
  auto tree = BuildTree<2>(points);
  std::cout << "Done building" << std::endl;
  arma::rowvec point{0, 0};
  bool in_tree = search<2>(tree, point);
}
