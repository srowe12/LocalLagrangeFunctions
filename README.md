[![Build Status](https://travis-ci.com/srowe12/LocalLagrangeFunctions.svg?branch=master)](https://travis-ci.com/srowe12/LocalLagrangeFunctions)

# LocalLagrangeFunctions
C++ Library for generation of Local Lagrange Functions and Radial Basis Function interpolants.

# Why use Radial Basis Functions?
Radial Basis Functions (RBFs) provide a approximation theoretically sound method for approximating functions of certain smoothness in low to high dimensional spaces. Most notably, RBFs are natural for working with scattered data. RBFs can be used for surface reconstruction, least squares approximation, as well as solving partial differential equations.

# Why use Local Lagrange Functions?
Local Lagrange Functions provide a nearly Lagrange ("Cardinal") basis for a set of points in R^n. If we wish to approximate a function defined on a subset of scattered points, X, in R^n, we can produce an interpolant using radial basis functions (RBFs).

A Lagrange function for a set of N centers `x_1...x_N` is a function `L_i` such that `L_i(x_j) = 1` if `i = j` and zero otherwise. Consequently, we can write out an interpolant for a function sampled at N scattered points as a sum `f(x_i) L_i(x)`. 

Forming the full set of Lagrange functions is computationally burdensome. We provide an initial capability of computing Local Lagrange functions, which are provably comparable to the full Lagrange function. The local Lagrange functions don't require all N points to construct; only log(N)^d points, where d is the dimension of the ambient space.

# Capabilities Provided

- Interpolation of many dimensional scattered data with a radial basis function interpolant
- Generation of a full set of Lagrange functions for easy interpolation of scattered, high dimensional data
- Generation of a full set of localized Lagrange functions for easy interpolation of scattered, high dimensional data

# Example usage

```cpp
 // Simulate some independent data (x,y) and dependent data (z) points
 std::vector<double> xmesh = mathtools::linspace<double>(0, 10, num_points);
 auto centers = mathtools::meshgrid<double>(xmesh, xmesh);
 const arma::vec x = centers.col(0);
 const arma::vec y = centers.col(1);
 const arma::vec z = sin(2*M_PI*x) % cos(2*M_PI*x); 

 size_t numcenters = centers.n_rows;
 // Make a Gaussian with scale parameter 1.0
 Gaussian<double> gaussian(1.0);
 // Create an RBF Interpolant with a Gaussian
 RadialBasisFunctionInterpolant<Gaussian<double>, 1> interpolant(gaussian, centers, data); 
 
 // Evaluate the interpolator on some new evaluation points
 const arma::mat evaluation_points = getSomeEvaluationPoints();
 
 // Evaluate to interpolate
 interpolated_data = interpolant.interpolate(evaluation_points);

```
