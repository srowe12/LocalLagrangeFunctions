#include <gtest/gtest.h>
#include <armadillo>

#include "../kdtree.h"

TEST(KdTreeTests, SimpleTest) {
	arma::mat points{{0,0},{1,1},{2,2}};

	auto tree = BuildTree<2>(points, 0);

    EXPECT_TRUE(tree != nullptr);
	tree->Print();
}