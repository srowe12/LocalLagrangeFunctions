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


 
template <size_t Dimension>
void findtuples(std::vector<std::array<size_t, Dimension>>& results, size_t sumtotal, const size_t position,
                                                      std::array<size_t, Dimension> currarray) {
  if ((sumtotal == 0) || (position == Dimension)) {
    return;
  }

  auto nextarray = currarray; // copy it for downstream use

  currarray[position] = sumtotal; // e.g. 2
  for (size_t pos = position; pos < Dimension; ++pos) {
    currarray[pos] = 0;
  }
  results.push_back(currarray);

  nextarray[position] = sumtotal -1;
  findtuples(results, sumtotal - 1, position + 1, nextarray);

  // Now decrement by 1, place it into that position, and move down
  
}

template <size_t Dimension>
void findtuples(std::vector<std::array<size_t, Dimension>>& results, size_t sumtotal, const size_t position) {
    std::array<size_t, Dimension> array;
    array.fill(0);
    findtuples<Dimension>(results, sumtotal, position, array);
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
