#ifndef LOCAL_LAGRANGE_POLYNOMIALS_H
#define LOCAL_LAGRANGE_POLYNOMIALS_H
#include <array>
#include <vector>
#include "math_tools.h"

template <size_t N>
using Tuple = std::array<double, N>;

template <size_t N>
using Tuples = std::vector<Tuple<N>>;
// Strategy: Lexicographic ordering implies we need to decrement iteratively
// down and shift
template <size_t Dimension,
          int TotalDegree,
          size_t Position,
          int CurrentTotal,
          int Value>
void decrement(Tuples<Dimension>& v, Tuple<Dimension> t) {
  if constexpr (Position == Dimension - 1) {
    t[Position] = TotalDegree - CurrentTotal;  // Grab what is left

    v.push_back(t);
    return;
  } else {
    t[Position] = Value;

    // Value should decrement from CurrentTotal to 0
    // Shift to the right and start over with the residual amount:
    // CurrentTotal = TotalDegree - (CurrentTotal + value)

    decrement<Dimension, TotalDegree, Position + 1, CurrentTotal + Value,
              TotalDegree - CurrentTotal - Value>(v, t);

    // Next, we will decrement the current value
    if constexpr (Value > 0) {
      decrement<Dimension, TotalDegree, Position, CurrentTotal, Value - 1>(v,
                                                                           t);
    } else {
      return;
    }
  }
}

template <size_t N, int TotalDegree>
Tuples<N> buildTuples() {
  Tuples<N> v;
  Tuple<N> t;
  t.fill(0);
  decrement<N, TotalDegree, 0, 0, TotalDegree>(v, t);
  return v;
}

template <size_t N, int TotalDegree>
void buildTuples(Tuples<N>& tuples) {
  Tuple<N> t;
  t.fill(0);
  decrement<N, TotalDegree, 0, 0, TotalDegree>(tuples, t);
}

template <size_t N, int TotalDegree>
void findPolynomialsUpToDegree(Tuples<N>& polynomials) {
  buildTuples<N, TotalDegree>(polynomials);
  if constexpr (TotalDegree >= 1) {
    findPolynomialsUpToDegree<N, TotalDegree - 1>(polynomials);
  } else {
    return;
  }
}

template <size_t Dimension = 2>
void applyPower(arma::mat& matrix,
                const arma::mat& points,
                const Tuple<Dimension>& powers,
                const int offset) {
  // Naively form the powers, refactor later as this is super inefficient
  arma::vec p = arma::pow(points.col(0), powers[0]);
  for (size_t i = 1; i < Dimension; ++i) {
    p %= arma::pow(points.col(i), powers[i]);
  }
  const size_t num_points = points.n_rows;
  matrix(arma::span(0, num_points - 1), offset) = p;

  matrix(offset, arma::span(0, num_points - 1)) = p.t();
}

template <size_t Dimension = 2>
double polynomialApply(const arma::vec& coefficients,
                       const arma::rowvec::fixed<Dimension>& p,
                       const std::vector<Tuple<Dimension>>& powers) {
  // Naively form the powers, refactor later as this is super inefficient
  double result = 0.0;
  int count = 0;
  for (const auto& tuple : powers) {
    double local_result = 1.0;
    for (size_t i = 0; i < Dimension; ++i) {
      local_result *= std::pow(p(i), tuple[i]);
    }
    result += coefficients(count) * local_result;
    ++count;
  }

  return result;
}

constexpr size_t factorial(size_t n) {
  if (n == 0) {
    return 1;
  } else {
    return n * factorial(n - 1);
  }
}

/**
 * @brief Computes the size of a polynomial basis of given degree in in
 * Dimension variables
 * @param[in] degree is the desired upper degree of the polynomials
 *
 * @return Returns the number of monomials needed to build the polynomial basis
 */
template <size_t Dimension>
constexpr inline size_t computePolynomialBasis(const size_t degree) {
  return factorial(Dimension + degree) /
         (factorial(degree) * factorial(Dimension));
}

template <size_t Dimension = 2, int Degree = 1>
void buildPolynomialMatrix(arma::mat& interpolation_matrix,
                           const arma::mat& points) {
  const size_t num_rows = interpolation_matrix.n_rows;
  const size_t num_cols = interpolation_matrix.n_cols;
  const size_t num_points = points.n_rows;

  const size_t num_polynomials = computePolynomialBasis<Dimension>(Degree);

  // Interpolation matrix for conditionally positive definite kernel is of the
  // form [A P; P^T 0]. Our goal is to fill in the matrix P. This form depends
  // on the degree and basis; The form of the matrix will in ascending order
  // from lowest degree constant polynomial to highest degree

  if (num_points + num_polynomials != num_rows) {
    throw std::runtime_error(
        "The size of the interpolation matrix is not "
        "equal to the number of points plus number of "
        "polynomial terms");
  }

  int column_offset = num_points;
  ///@todo srowe: Should degree = 0 or 1? Should we shift colun offset?
  // Given dim, and deg, find all subsets (x_1,x_2...x_dim) sums to deg
  // For degree we have x^i*y^degree-i
  // xyz poly would be for degree 1: x y z
  // For degree 2 we would have x^2 + xy + xz + y^2 + yz + z^2;
  // For degree 3 we would have x^3 + x^2 y + x^2z + x
  // (3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (0,3,0), (0,2,1), (0,1,2),
  // (0,)
  Tuples<Dimension> exponent_tuples;
  findPolynomialsUpToDegree<Dimension, Degree>(
      exponent_tuples);  // Loop over exponent tuples placing them into the
                         // matrix
  for (const auto& tuple : exponent_tuples) {
    applyPower(interpolation_matrix, points, tuple, column_offset);
    ++column_offset;
  }
}

#endif  // LOCAL_LAGRANGE_POLYNOMIALS_H