#ifndef LOCAL_LAGRANGE_ASSEMBLER_HDR
#define LOCAL_LAGRANGE_ASSEMBLER_HDR

#include <armadillo>
#include <array>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "local_lagrange.h"
#include <kdtree/kdtree.h>

namespace local_lagrange {

template <size_t Dimension = 2, typename Kernel = ThinPlateSpline<Dimension>>
class LocalLagrangeAssembler {
public:
  LocalLagrangeAssembler(const arma::mat &centers, const double radius)
      : centers_(centers), radius_(radius),
        kdtree_root_(BuildTree<Dimension>(centers_)) {}

  unsigned int findLocalIndex(const arma::mat &local_centers,
                              unsigned int index) {

    const arma::rowvec center = centers_.row(index);

    // Implement naive algorithm here. Upgrade later.
    // Machine precision equality errors possible.
    unsigned int local_index = 0;
    const size_t num_vectors = local_centers.n_rows;

    for (size_t i = 0; i < num_vectors; ++i) {
      auto matching = arma::all(center == local_centers.row(i));

      if (matching) {
        local_index = i;
        break;
      }
    }

    ///@todo srowe: If we fail to find this index, we return 0, which is wrong
    return local_index;
  }

  LocalLagrange<Dimension>
  generateLocalLagrangeFunction(const unsigned int index) {

    // Let's query the data via kdtree
    const arma::rowvec local_point = centers_.row(index);
    const std::vector<arma::rowvec> local_centers_v =
        RadiusQuery<Dimension>(kdtree_root_, local_point, radius_);
    // Stupidly make an arma mat from this
    arma::mat local_centers(local_centers_v.size(), Dimension);
    int i = 0;
    for (const auto &row : local_centers_v) {
      local_centers.row(i) = row;
      ++i;
    }

    const size_t local_index = findLocalIndex(local_centers, index);
    LocalLagrange<Dimension, Kernel> llf(local_centers, local_index);

    return llf;
  }

  unsigned int num_centers() const { return num_centers_; }
  double scale_factor() const { return scale_factor_; }
  double mesh_norm() const { return mesh_norm_; }
  double ball_radius() const { return ball_radius_; }

  void setScale_factor(double scale_factor) {
    scale_factor_ = scale_factor;
    updateBallRadius();
  }
  void setMesh_norm(double mesh_norm) {
    mesh_norm_ = mesh_norm;
    updateBallRadius();
  }
  void setCenters(const arma::mat &centers) {
    // Assumes size of centers_x and centers_y are the same
    centers_ = centers;
    num_centers_ = centers_.n_rows;
  }

  void setNum_local_centers(const unsigned int num_local_centers) {
    num_local_centers_ = num_local_centers;
  }

private:
  void updateBallRadius() {
    ball_radius_ = scale_factor_ * mesh_norm_ * abs(log(mesh_norm_));
  }

  unsigned int num_centers_;
  double scale_factor_; // We use ball_radius =
                        // scale_factor*mesh_norm*abs(log(mesh_norm));
  arma::mat centers_;   // N x d, with N points and d dimensions.
  unsigned int num_local_centers_; // How many local centers should we find?
  double radius_;
  double mesh_norm_;
  double ball_radius_;
  std::shared_ptr<Node<Dimension>> kdtree_root_;
};

} // namespace local_lagrange
#endif // LOCAL_LAGRANGE_ASSEMBLER_HDR
