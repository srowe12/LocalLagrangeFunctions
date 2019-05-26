#ifndef LOCAL_LAGRANGE_INTERPOLANT_H
#define LOCAL_LAGRANGE_INTERPOLANT_H

#include "local_lagrange.h"
#include "local_lagrange_assembler.h"

namespace local_lagrange {
// Provides ability to interpolate functions with a collection of Local Lagrange
// Functions
template <size_t Dimension, typename Kernel = ThinPlateSpline<Dimension>>
class LocalLagrangeEnsemble {
public:
  LocalLagrangeEnsemble(const std::vector<LocalLagrange<Dimension>>& llfs)
      : m_llfs(llfs) {}

  LocalLagrangeEnsemble(std::vector<LocalLagrange<Dimension>>&& llfs)
      : m_llfs(std::move(llfs)) {}

  std::vector<LocalLagrange<Dimension>> localLagrangeFunctions() const {
    return m_llfs;
  }

private:
  std::vector<LocalLagrange<Dimension>>
      m_llfs;  // Vector of Local Lagrange Functions
};

template <size_t Dimension, typename Kernel = ThinPlateSpline<Dimension>>
LocalLagrangeEnsemble<Dimension> buildLocalLagrangeFunctions(
    const arma::mat& centers,
    const double radius) {
  // Instantiate a LocalLagrangeAssembler

  LocalLagrangeAssembler<Dimension> assembler(centers, radius);

  std::vector<LocalLagrange<Dimension>> llfs;
  size_t num_centers = centers.n_rows;
  llfs.reserve(num_centers);

  for (size_t i = 0; i < num_centers; ++i) {
    llfs.emplace_back(assembler.generateLocalLagrangeFunction(i));
  }

  return LocalLagrangeEnsemble<Dimension>(llfs);
}

template <size_t Dimension, typename Kernel = ThinPlateSpline<Dimension>>
class LocalLagrangeInterpolant {
public:
  LocalLagrangeInterpolant(const LocalLagrangeEnsemble<Dimension>& lle,
                           const arma::vec& sampled_function)
      : m_llfs(lle.localLagrangeFunctions()),
        m_sampled_function(sampled_function) {
    for (size_t i = 0; i < sampled_function.size(); ++i) {
      m_llfs[i].scaleCoefficients(sampled_function(i));  // Multiply each LLF
                                                         // by the sampled
                                                         // center value
                                                         // corresponding to
                                                         // it
    }
  }

  double operator()(const arma::rowvec& point) const {
    double result = 0.0;
    for (const auto& llf : m_llfs) {
      result += llf(point);
    }

    return result;
  }

private:
  std::vector<LocalLagrange<Dimension, Kernel>> m_llfs;
  arma::vec m_sampled_function;
};

}  // end namespace local_lagrange

#endif  // LOCAL_LAGRANGE_INTERPOLANT_H
