#include "local_lagrange_interpolant.h"
namespace local_lagrange {

LocalLagrangeEnsemble buildLocalLagrangeFunctions(const arma::vec &centers_x,
                                                  const arma::vec &centers_y,
                                                  size_t num_local_centers) {
  // Instantiate a LocalLagrangeAssembler

  LocalLagrangeAssembler assembler(centers_x, centers_y, num_local_centers);

  std::vector<LocalLagrange> llfs;
  size_t num_centers = centers_x.size();
  llfs.reserve(num_centers);

  for (size_t i = 0; i < num_centers; ++i) {
    llfs.emplace_back(assembler.generateLocalLagrangeFunction(i));
  }

  return LocalLagrangeEnsemble(llfs);
}

} // end namespace local_lagrange
