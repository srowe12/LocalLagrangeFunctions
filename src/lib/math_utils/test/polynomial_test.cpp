
#include <math_utils/polynomials.h>
#include <catch2/catch.hpp>

TEST_CASE("BuildTuples") {
  const auto results = buildTuples<3, 4>();

  Tuples<3> expected;
  expected.push_back({4, 0, 0});
  expected.push_back({3, 1, 0});
  expected.push_back({3, 0, 1});
  expected.push_back({2, 2, 0});
  expected.push_back({2, 1, 1});
  expected.push_back({2, 0, 2});
  expected.push_back({1, 3, 0});
  expected.push_back({1, 2, 1});
  expected.push_back({1, 1, 2});
  expected.push_back({1, 0, 3});
  expected.push_back({0, 4, 0});
  expected.push_back({0, 3, 1});
  expected.push_back({0, 2, 2});
  expected.push_back({0, 1, 3});
  expected.push_back({0, 0, 4});

  REQUIRE(expected.size() == results.size());
  REQUIRE(expected == results);

  // (4,0,0)
  // (3, 1, 0)
  // (3, 0, 1)
  // (2, 2, 0)
  // (2, 1, 1)
  // (2, 0, 2);
  // (1, 3, 0);
  // (1, 2, 1);
  // (1, 1, 2);
  // (1, 0, 3);
  // (0, 4 , 0);
  // (0, 3, 1);
  // (0, 2, 2);
  // (0, 1, 3);
  // (0, 0, 4);
}


template <size_t Dimension>
bool compareTuples(const std::vector<Tuple<Dimension>>& t1,
                   const std::vector<Tuple<Dimension>>& t2) {
  if (t1.size() != t2.size()) {
    return false;
  }

  bool success = true;
  for (const auto& i : t1) {
    // Check that i is in t2;
    success &= std::find(t2.begin(), t2.end(), i);
  }

  return success;
}
TEST_CASE("FindTuplesDim3Degree1") {
  // Expect (1,0,0), (0,1,0), (0,0,1)

  auto results = buildTuples<3,1>();

  Tuple<3> expected0 = {1, 0, 0};
  Tuple<3> expected1 = {0, 1, 0};
  Tuple<3> expected2 = {0, 0, 1};

  std::vector<Tuple<3>> expected{expected0, expected1, expected2};
  REQUIRE(expected == results);
  REQUIRE(results.size() == 3);
}

TEST_CASE("FindtuplesDim3Degree2") {
  // Expect (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,2,0)
  auto results = buildTuples<3,2>();
  std::vector<Tuple<3>> expected{{2, 0, 0}, {1, 1, 0}, {1, 0, 1},
                                 {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};

  REQUIRE(expected == results);
  REQUIRE(results.size() == expected.size());
}


TEST_CASE("TestApplyPower") {
  Tuple<3> tuple{1, 2, 3};

  const arma::mat points{{2, 2, 2}, {1, 1, 1}, {1, 2, 3}, {2, 3, 1}};
  int num_polys = 1;
  arma::mat matrix = arma::zeros(4 + num_polys, 4 + num_polys);
  int offset = 4;
  applyPower<3>(matrix, points, tuple, offset);
  const arma::vec& x = points.col(0);
  const arma::vec& y = points.col(1);
  const arma::vec& z = points.col(2);
  const arma::vec expected = x % y % y % z % z % z;
  const arma::vec computed = matrix.col(4).rows(0, 3);
  const arma::vec computed_row = matrix.row(4).cols(0, 3).t();

  const double error = arma::abs(expected - computed).max();
  REQUIRE(error <= 1e-8);

  const double row_error = arma::abs(expected - computed_row).max();
  REQUIRE(error <= 1e-8);
}

TEST_CASE("TestBuildPolyMatrix") {
  const arma::mat points{{2, 2, 2}, {1, 1, 1}, {1, 2, 3}, {2, 3, 1}};
  int num_polys = 4;
  arma::mat matrix = arma::zeros(4 + num_polys, 4 + num_polys);
  int offset = 4;
  buildPolynomialMatrix<3, 1>(matrix, points);
  const arma::vec& x = points.col(0);
  const arma::vec& y = points.col(1);
  const arma::vec& z = points.col(2);

  arma::mat expected = arma::ones(4, 4);
  expected.col(0) = x;
  expected.col(1) = y;
  expected.col(2) = z;

  // Form should be [A P ; P^T 0] so check P upper part should match expected
  const arma::mat upper_right = matrix(arma::span(0, 3), arma::span(4, 7));

  const double error = (expected - upper_right).max();
  REQUIRE(error <= 1e-8);
  // Then we shoudl have P^T in bottom left
  const arma::mat lower_left = matrix(arma::span(4, 7), arma::span(0, 3));
  const double error_lower_left = (expected.t() - lower_left).max();
  REQUIRE(error_lower_left <= 1e-8);
  // Then zeros in lower right block
  const arma::mat lower_right = matrix(arma::span(4, 7), arma::span(4, 7));
  const double error_lower_right = (lower_right).max();
  REQUIRE(error_lower_right <= 1e-8);
}

TEST_CASE("TestApplyPolynomial") {
  // Define polynomial 1 + 2x + 3y + 4z + 5x^2 + 6xy + 7xz + 8y^2 + 9yz + 10z^2;
  // x = 1, y =2 , z = 3;
  // p(x,y,z) = 1 + 2 + 6 + 12 + 5 + 12 + 21 + 32 + 54 + 90

  std::vector<Tuple<3>> powers;
  powers.push_back({0, 0, 0});
  powers.push_back({1, 0, 0});
  powers.push_back({0, 1, 0});
  powers.push_back({0, 0, 1});
  powers.push_back({2, 0, 0});
  powers.push_back({1, 1, 0});
  powers.push_back({1, 0, 1});
  powers.push_back({0, 2, 0});
  powers.push_back({0, 1, 1});
  powers.push_back({0, 0, 2});

  arma::vec coefficients{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  arma::rowvec::fixed<3> p{1, 2, 3};
  const double result = polynomialApply<3>(coefficients, p, powers);
  const double expected = 1 + 2 + 6 + 12 + 5 + 12 + 21 + 32 + 54 + 90;
  REQUIRE(expected == result);  
}

TEST_CASE("FactorialTestZero") {
  size_t n = 0;
  REQUIRE(1 == factorial(n));
}

TEST_CASE("FactorialTestOne") {
  REQUIRE(1 == factorial(1));
}

TEST_CASE("FactorialTestLarger") {
  REQUIRE(4 * 3 * 2 == factorial(4));
}

TEST_CASE("PolynomialSize") {
  size_t degree = 3;
  // basis is 1  ,x ,y, x2, xy, y2, x3 ,x2y, xy2, y3
  size_t size_poly_basis = computePolynomialBasis<2>(degree);
  REQUIRE(10 == size_poly_basis);
}