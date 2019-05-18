
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <math_utils/polynomials.h>


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