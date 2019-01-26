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
  const double radius = .1 * .1;
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
  const double radius = .5 * .5;
  arma::rowvec point{.75, .75};

  const std::vector<arma::rowvec> found_points =
      RadiusQuery<2>(tree, point, radius);

  ASSERT_EQ(2, found_points.size());

  arma::rowvec expected_first_point{.5, .5};
  arma::rowvec expected_second_point{1, 1};

  EXPECT_TRUE(inVector(found_points, expected_first_point));
  EXPECT_TRUE(inVector(found_points, expected_second_point));
}

TEST(KdtreeTests, RadiusQueryMultiplePointsBigger) {
  arma::mat points{{0, 0}, {1, 1}, {0, 1}, {1, 0}, {.5, .5}};

  auto tree = BuildTree<2>(points);
  const double radius = .8 * .8;
  arma::rowvec point{.75, .75};

  const std::vector<arma::rowvec> found_points =
      RadiusQuery<2>(tree, point, radius);

  ASSERT_EQ(4, found_points.size());

  arma::rowvec expected_first_point{.5, .5};
  arma::rowvec expected_second_point{1, 1};
  arma::rowvec expected_third_point{1, 0};
  arma::rowvec expected_fourth_point{0, 1};

  EXPECT_TRUE(inVector(found_points, expected_first_point));
  EXPECT_TRUE(inVector(found_points, expected_second_point));
  EXPECT_TRUE(inVector(found_points, expected_third_point));
  EXPECT_TRUE(inVector(found_points, expected_fourth_point));
}

bool CompareSets(const std::vector<arma::rowvec> &a,
                 const std::vector<arma::rowvec> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  bool return_val = true;
  for (size_t i = 0; i < a.size(); ++i) {
    arma::rowvec p = a[i];
    bool compare = false;
    for (size_t j = 0; j < b.size(); ++j) {
      if (arma::norm(p - b[j])) {
        compare = true;
      }
    }
    return_val &= compare;
  }
  return return_val;
}

TEST(KdTreeTest, RadiusQueryMultiplePointsBiggerIncludingPoint) {

  size_t num_points = 50;

  auto xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto points = mathtools::meshgrid<double>(xmesh, xmesh);
  arma::rowvec p1 = points.row(0);
  arma::rowvec p2 = points.row(1);
  p1.print("p1");
  p2.print("p2");
  std::cout << "The computed distance between p1 and p2 is"
            << mathtools::computeDistance<1>(p1, p2) << "\n";
  auto tree = BuildTree<2>(points);
  if (!tree) {
    std::cout << "Tree is nulL!" << std::endl;
  }
  const double radius = .1 * .1;

  bool in_tree = search<2>(tree, p1);
  if (in_tree) {
    std::cout << "p1 found "
              << "\n";
  } else {
    std::cout << "p1 not found\n";
  }

  const auto found_points = RadiusQuery<2>(tree, p1, radius);

  std::cout << "Size of found_oints " << found_points.size() << "\n";

  int num_rows = points.n_rows;
  std::vector<arma::rowvec> naive;
  for (int i = 0; i < num_rows; ++i) {
    double dist = mathtools::computeDistance<1>(p1, points.row(i));
    if (dist <= .1 * .1) {
      naive.push_back(points.row(i));
    }
  }
  std::cout << "Size of naive " << naive.size() << "\n";

  for (const auto &p : found_points) {
    p.print("");
  }

  EXPECT_EQ(naive.size(), found_points.size());

  EXPECT_TRUE(CompareSets(found_points, naive));
}
