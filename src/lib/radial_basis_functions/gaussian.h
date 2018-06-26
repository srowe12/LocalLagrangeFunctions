#ifndef LOCAL_LAGRANGE_GAUSSIAN_H
#define LOCAL_LAGRANGE_GAUSSIAN_H

template <double ScaleParameter>
struct Gaussian {

   operator()(const double r_squared) {
       return std::exp(-ScaleParameter*r_squared);
   }
};
#endif // LOCAL_LAGRANGE_GAUSSIAN_H

