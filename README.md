[![Build Status](https://travis-ci.com/srowe12/LocalLagrangeFunctions.svg?branch=master)](https://travis-ci.com/srowe12/LocalLagrangeFunctions)

# LocalLagrangeFunctions
C++ Library for generation of Local Lagrange Functions and Radial Basis Function interpolants.

# Why use Radial Basis Functions?
Radial Basis Functions (RBFs) provide a approximation theoretically sound method for approximating functions of certain smoothness in low to high dimensional spaces. Most notably, RBFs are natural for working with scattered data. RBFs can be used for surface reconstruction, least squares approximation, as well as solving partial differential equations.

# Why use Local Lagrange Functions?
Local Lagrange Functions provide a nearly Lagrange ("Cardinal") basis for a set of points in R^n. If we wish to approximate a function defined on a subset of scattered points, X, in R^n, we can produce an interpolant using radial basis functions (RBFs).

A Lagrange function for a set of N centers x_1...x_N is a function L_i such that L_i(x_j) = 1 if i = j and zero otherwise. Consequently, we can write out an interpolant for a function sampled at N scattered points as sum f(x_i) L_i(x). 

Forming the full set of Lagrange functions is computationally burdensome. We provide an initial capability of computing Local Lagrange functions, which are provably comparable to the full Lagrange function. The local Lagrange functions don't require all N points to construct; only log(N)^d points, where d is the dimension of the ambient space.

# Capabilities Provided

