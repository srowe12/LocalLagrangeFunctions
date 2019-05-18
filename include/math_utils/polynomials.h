#ifndef LOCAL_LAGRANGE_POLYNOMIALS_H
#define LOCAL_LAGRANGE_POLYNOMIALS_H
#include <array>
#include <iostream>
#include <vector>
template <size_t N> using Tuple = std::array<double, N>;

template <size_t N> using Tuples = std::vector<Tuple<N>>;

// Strategy: Lexicographic ordering implies we need to decrement iteratively
// down and shift

template <size_t N> void print(Tuple<N> t) {
  for (auto &i : t) {
    std::cout << i << " , ";
  }
  std::cout << "\n";
}
template <size_t Dimension, int TotalDegree, size_t Position, int CurrentTotal,
          int Value>
void decrement(Tuples<Dimension> &v, Tuple<Dimension> t) {

  if constexpr (Position == Dimension - 1) {

    t[Position] = TotalDegree - CurrentTotal; // Grab what is left
    std::cout << "Placing in the remainder\n: CAUSE " << Position << " with "
              << Value << "\n";
    print(t);
    v.push_back(t);
    return;
  } else {
    t[Position] = Value; // Value should decrement from CurrentTotal to 0
    // Shift to the right
    std::cout << "Current Position " << Position
              << " and CurrentTotal = " << CurrentTotal
              << " and Value = " << Value << "\n";

    decrement<Dimension, TotalDegree, Position + 1, CurrentTotal + Value,
              TotalDegree - CurrentTotal - Value>(v, t);

    std::cout << "Completed Iterative Inner Loop \n";
    // Next, we will decrement the current value
    if constexpr (Value > 0) {
      decrement<Dimension, TotalDegree, Position, CurrentTotal, Value - 1>(v,
                                                                           t);
    } else {
      // v.push_back(t);
      return;
    }
  }
  // For Value = TotalDegree - CurrentTotal to 0,
  // tuple[position] = ThatValue
  // decrement<Dimension, TotalDegree, Position+1, CurrentTotal, Value>;
  // decrement<Dimension, TotalDegree, Position, CurrentTotal+1
  // Should do...
  // (4,0,0)
  // (3, 1, 0)
  // (3, 0, 1)
  // (2, 2, 0)
  // (2, 1, 1)
  // (2, 0, 2);
  // (1, 3, 0);
  // (1, 2, 1);
  // (1, 1, 2);
  // (1, 0, 3);
  // (0, 4 , 0);
  // (0, 3, 1);
  // (0, 2, 2);
  // (0, 1, 3);
  // (0, 0, 4);
}

template <size_t N, int TotalDegree> Tuples<N> buildTuples() {
  Tuples<N> v;
  Tuple<N> t;
  t.fill(0);
  decrement<N, TotalDegree, 0, 0, 4>(v, t);
  return v;
}

#endif // LOCAL_LAGRANGE_POLYNOMIALS_H