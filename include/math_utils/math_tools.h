#ifndef MATH_UTILS_HDR
#define MATH_UTILS_HDR

#include <armadillo>
#include <array>
#include <fstream>
#include <iterator>
#include <math.h>
#include <string>
#include <vector>

namespace mathtools {

// Inefficient approach: Make a {0,1,...degree}^dimension cube and iterate
// through them all.
template <size_t Dimension>
using Tuple = std::array<int, Dimension>;

template <size_t Dimension> 
int sum(const Tuple<Dimension> &t) {
  int result = 0;
  for (size_t i = 0; i < Dimension; ++i) {
    result += t[i];
  }
  return result;
}

template <size_t Dimension, size_t current_dimension>
void findtuples(const int degree, Tuple<Dimension> tuple,
                std::vector<Tuple<Dimension>> &results) {
  // Recursively walk back

  for (int i = 0; i <= degree; ++i) {
    // If we are in the last position, we are done, so test it
    if constexpr (Dimension-1 == (current_dimension)) {
      tuple[current_dimension] = i;
      if (sum(tuple) == degree) {
        std::cout << "Calling Push Back with i = " << i << " and current dim  = " << current_dimension << "\n";
        results.push_back(tuple);
      }
    } else {
      Tuple<Dimension> candidate = tuple; // copy the current result thats
                                          // filled, and place an i in Dimension
                                          // slot
      candidate[current_dimension] = i;

      findtuples<Dimension, current_dimension + 1>(degree, candidate, results);
    }
  }
}

template <size_t Dimension>
std::vector<Tuple<Dimension>> findtuples(const int degree) {
  std::vector<Tuple<Dimension>> results;
  for (int i = 0; i <= degree; ++i) {
    Tuple<Dimension> candidate;
    candidate.fill(0);
    candidate[0] = i;
    findtuples<Dimension, 1>(degree, candidate, results);
  }

  return results;
}

template <size_t Dimension=2>
void applyPower(arma::mat& matrix, const arma::mat& points, const Tuple<Dimension>& powers, const int offset ) {
  // Naively form the powers, refactor later as this is super inefficient
  arma::vec p = arma::pow(points.col(0), powers[0]);
  for (size_t i = 1; i < Dimension; ++i) {
    p %= arma::pow(points.col(i), powers[i]);
  }
  
  const size_t num_points = points.n_rows;
  std::cout << "Applying to column offset\n";
  p.print("P");
  matrix.print("Matrix");
  matrix(arma::span(0,num_points-1), offset) = p;
    std::cout << "Applying to row offset\n";

  matrix(offset, arma::span(0, num_points-1)) = p.t();
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


template <size_t Dimension =2, int Degree =1>
void buildPolynomialMatrix(arma::mat& interpolation_matrix, const arma::mat& points)  {
    const size_t num_rows = interpolation_matrix.n_rows;
    const size_t num_cols = interpolation_matrix.n_cols;
    const size_t num_points = points.n_rows;

    const size_t num_polynomials = computePolynomialBasis<Dimension>(Degree);

    // Interpolation matrix for conditionally positive definite kernel is of the form [A P; P^T 0]. 
    // Our goal is to fill in the matrix P. This form depends on the degree and basis;
    // The form of the matrix will in ascending order from lowest degree constant polynomial
    // to highest degree

    if (num_points + num_polynomials != num_rows) {
      throw std::runtime_error("The size of the interpolation matrix is not equal to the number of points plus number of polynomial terms");
    }

    interpolation_matrix(arma::span(0,num_points-1), num_points) = 1.0; 
    int column_offset = num_points;
    for (size_t degree = 0; degree < Degree; ++degree) {
        // Given dim, and deg, find all subsets (x_1,x_2...x_dim) sums to deg
        // For degree we have x^i*y^degree-i
        // xyz poly would be for degree 1: x y z
        // For degree 2 we would have x^2 + xy + xz + y^2 + yz + z^2;
        // For degree 3 we would have x^3 + x^2 y + x^2z + x
        // (3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (0,3,0), (0,2,1), (0,1,2), (0,)
        auto exponent_tuples = findtuples<Dimension>(degree);

        // Loop over exponent tuples placing them into the matrix
        for (const auto& tuple: exponent_tuples) {
          applyPower(interpolation_matrix, points, tuple, column_offset);
          ++column_offset;
        }
    }
  }


template <size_t Dimension, size_t Coordinate>
constexpr inline double
computeLengthSquared(const arma::rowvec::fixed<Dimension> &v) {
  if constexpr (Coordinate == 0) {
    return v(0) * v(0);
  } else {
    return v(Coordinate) * v(Coordinate) +
           computeLengthSquared<Dimension, Coordinate - 1>(v);
  }
}

template <size_t Dimension>
constexpr inline double
computeLengthSquared(const arma::rowvec::fixed<Dimension> &v) {
  if constexpr (Dimension > 1) {
    return computeLengthSquared<Dimension, Dimension - 1>(v);
  } else {
    return v(0) * v(0);
  }
}

///@todo srowe: These need to be marked constexpr I think
template <size_t Dimension, size_t Coordinate>
constexpr inline double
computeSquaredDistance(const arma::rowvec::fixed<Dimension> &v1,
                       const arma::rowvec::fixed<Dimension> &v2) {

  if constexpr (Coordinate == 0) {
    return (v1(0) - v2(0)) * (v1(0) - v2(0));
  } else {
    return (v1(Coordinate) - v2(Coordinate)) *
               (v1(Coordinate) - v2(Coordinate)) +
           computeSquaredDistance<Dimension, Coordinate - 1>(v1, v2);
  }
}

template <size_t Dimension>
constexpr inline double
computeSquaredDistance(const arma::rowvec::fixed<Dimension> &v1,
                       const arma::rowvec::fixed<Dimension> &v2) {
  if constexpr (Dimension > 1) {
    return computeSquaredDistance<Dimension, Dimension - 1>(v1, v2);
  } else {
    return (v1(0) - v2(0)) * (v1(0) - v2(0));
  }
}

template <size_t Coordinate>
double computeDistance(const arma::rowvec &p1, const arma::rowvec &p2) {
  return computeDistance<Coordinate - 1>(p1, p2) +
         (p1(Coordinate) - p2(Coordinate)) * (p1(Coordinate) - p2(Coordinate));
}

template <>
double computeDistance<0>(const arma::rowvec &p1, const arma::rowvec &p2) {
  return (p1(0) - p2(0)) * (p1(0) - p2(0));
}

template <size_t Coordinate>
inline double computeDistance(const size_t row, const size_t col,
                              const arma::mat &points) {
  return (points(row, Coordinate) - points(col, Coordinate)) *
             (points(row, Coordinate) - points(col, Coordinate)) +
         computeDistance<Coordinate - 1>(row, col, points);
}

template <>
inline double computeDistance<0>(const size_t row, const size_t col,
                                 const arma::mat &points) {
  return (points(row, 0) - points(col, 0)) * (points(row, 0) - points(col, 0));
}

template <size_t Coordinate>
inline double computePointDistance(const size_t i, const size_t j,
                                   const arma::mat &points,
                                   const arma::mat &other_points) {
  return (points(i, Coordinate) - other_points(j, Coordinate)) *
             (points(i, Coordinate) - other_points(j, Coordinate)) +
         computePointDistance<Coordinate - 1>(i, j, points, other_points);
}

template <>
inline double computePointDistance<0>(const size_t i, const size_t j,
                                      const arma::mat &points,
                                      const arma::mat &other_points) {
  return (points(i, 0) - other_points(j, 0)) *
         (points(i, 0) - other_points(j, 0));
}

///@todo srowe: Probably eliminate these in favor of Armadillo

template <typename T>
std::vector<T> linspace(T a, T b, unsigned int num_points) {
  T step_size = (b - a) / num_points;
  std::vector<T> points(num_points + 1); // One extra point to get last value.
  size_t counter = 0;
  for (auto it = points.begin(); it != points.end(); ++it) {
    *it = a + step_size * counter;
    counter++;
  }
  return points;
}

template <typename T>
arma::mat meshgrid(const std::vector<T> &xpoints,
                   const std::vector<T> &ypoints) {

  size_t num_points_x = xpoints.size();
  size_t num_points_y = ypoints.size();
  size_t num_points = num_points_x * num_points_y;
  std::vector<T> X(num_points);
  std::vector<T> Y(num_points);
  T xval;
  T yval;
  for (size_t row = 0; row < num_points_x; ++row) {
    xval = xpoints[row];
    for (size_t col = 0; col < num_points_y; ++col) {
      yval = ypoints[col];
      X[row * num_points_y + col] = xval;
      Y[row * num_points_y + col] = yval;
    }
  }
  arma::mat return_points(num_points, 2);
  return_points.col(0) = arma::conv_to<arma::vec>::from(X);
  return_points.col(1) = arma::conv_to<arma::vec>::from(Y);
  return return_points;
}

template <typename T>
void write_vector(std::vector<T> &vec, std::string file_name) {
  std::ofstream output_file(file_name);
  std::ostream_iterator<T> output_iterator(output_file, "\n");
  std::copy(vec.begin(), vec.end(), output_iterator);
}
} // namespace mathtools
#endif // MATH_UTILS_HDR
