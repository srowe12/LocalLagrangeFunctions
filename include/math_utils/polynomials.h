#ifndef LOCAL_LAGRANGE_POLYNOMIALS_H
#define LOCAL_LAGRANGE_POLYNOMIALS_H
#include <array>
#include <vector>
template <size_t N> using Tuple = std::array<double, N>;

template <size_t N> using Tuples = std::vector<Tuple<N>>;

// Strategy: Lexicographic ordering implies we need to decrement iteratively
// down and shift
template <size_t Dimension, int TotalDegree, size_t Position, int CurrentTotal,
          int Value>
void decrement(Tuples<Dimension> &v, Tuple<Dimension> t) {

  if constexpr (Position == Dimension - 1) {

    t[Position] = TotalDegree - CurrentTotal; // Grab what is left

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

template <size_t N, int TotalDegree> Tuples<N> buildTuples() {
  Tuples<N> v;
  Tuple<N> t;
  t.fill(0);
  decrement<N, TotalDegree, 0, 0, 4>(v, t);
  return v;
}

#endif // LOCAL_LAGRANGE_POLYNOMIALS_H