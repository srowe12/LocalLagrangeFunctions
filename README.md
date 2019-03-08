# LocalLagrangeFunctions
C++ Library for generation of Local Lagrange Functions

# Why use Local Lagrange Functions?
Local Lagrange Functions provide a nearly Lagrange ("Cardinal") basis for a set of points in R^n. If we wish to approximate a function defined on a subset of scattered points, X, in R^n, we can produce an interpolant using radial basis functions (RBFs).

A Lagrange function for a set of N centers x_1...x_N is a function L_i such that L_i(x_j) = 1 if i = j and zero otherwise. Consequently, we can write out an interpolant for a function sampled at N scattered points as sum f(x_i) L_i(x). 

Forming the full set of Lagrange functions is computationally burdensome. We provide an initial capability of computing Local Lagrange functions, which are provably comparable to the full Lagrange function. The local Lagrange functions don't require all N points to construct; only log(N)^d points, where d is the dimension of the ambient space.
