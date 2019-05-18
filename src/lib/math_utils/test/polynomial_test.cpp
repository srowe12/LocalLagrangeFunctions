
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <math_utils/polynomials.h>
#include <stdio.h>

template <size_t N> void print(const Tuples<N> &t) {
  for (auto &tuple : t) {
    for (auto &i : tuple) {
      std::cout << i << ",";
    }
    std::cout << "\n";
  }
}
TEST_CASE("BuildTuples") {
  auto results = buildTuples<3, 4>();
  print(results);
  REQUIRE(1 == 0);
}