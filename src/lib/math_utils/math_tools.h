#ifndef MATH_UTILS_HDR
#define MATH_UTILS_HDR
#include <math.h>
#include <vector>
#include <array>

namespace mathtools {

template <typename T>
std::vector<T> linspace(T a, T b, unsigned int num_points){
   T step_size = (b-a)/num_points; 
   std::vector<T> points(num_points+1); //One extra point to get last value.
   size_t counter = 0;
   for (auto it = points.begin(); it!= points.end(); ++it){
      *it = a+step_size*counter;
      counter++;
   }
   return points;
}

} //namespace mathtools
#endif //MATH_UTILS_HDR
