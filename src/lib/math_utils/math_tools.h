#ifndef MATH_UTILS_HDR
#define MATH_UTILS_HDR
#include <array>
#include <fstream>
#include <iterator>
#include <math.h>
#include <string>
#include <vector>

namespace mathtools {
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
std::array<std::vector<T>, 2> meshgrid(const std::vector<T> &xpoints,
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
  std::array<std::vector<T>, 2> return_points{{X, Y}};
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
