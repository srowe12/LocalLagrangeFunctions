#ifndef LOCAL_LAGRANGE_INTERPOLANT_H
#define LOCAL_LAGRANGE_INTERPOLANT_H

#include "local_lagrange.h"
#include "local_lagrange_assembler.h"

namespace local_lagrange {
// Provides ability to interpolate functions with a collection of Local Lagrange
// Functions
class LocalLagrangeEnsemble {
public:
  LocalLagrangeEnsemble(const std::vector<LocalLagrange> &llfs)
      : m_llfs(llfs) {}

  LocalLagrangeEnsemble(std::vector<LocalLagrange> &&llfs)
      : m_llfs(std::move(llfs)) {}

  std::vector<LocalLagrange> localLagrangeFunctions() const { return m_llfs; }

private:
  std::vector<LocalLagrange> m_llfs; // Vector of Local Lagrange Functions
};

LocalLagrangeEnsemble buildLocalLagrangeFunctions(const arma::vec &centers_x,
                                                  const arma::vec &centers_y,
                                                  size_t num_local_centers);

class LocalLagrangeInterpolant {
public:
  LocalLagrangeInterpolant(const LocalLagrangeEnsemble &lle,
                           const arma::vec &sampled_function)
      : m_llfs(lle.localLagrangeFunctions()),
        m_sampled_function(sampled_function) {
    std::cout << "Beginning stuff" << std::endl;
    for (size_t i = 0; i < sampled_function.size(); ++i) {
      m_llfs[i].scaleCoefficients(sampled_function(i)); // Multiply each LLF
                                                        // by the sampled
                                                        // center value
                                                        // corresponding to
                                                        // it
    }
  }

  double operator()(const double x, const double y) const {
    double result = 0.0;
    for (const auto &llf : m_llfs) {
      result += llf(x, y);
    }

    return result;
  }

private:
  std::vector<LocalLagrange> m_llfs;
  arma::vec m_sampled_function;
};

} // end namespace local_lagrange

#endif // LOCAL_LAGRANGE_INTERPOLANT_H
