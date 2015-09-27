#include "../local_lagrange.h"
#include <gtest/gtest.h>
#include <stdio.h>
#include <utility>
#include <string>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>

#include <boost/geometry/index/rtree.hpp>

#include "math_tools.h"

TEST(IntegrationTest,BuildAnLLF){

  size_t num_points = 50;

  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  std::array<std::vector<double>, 2> centers =
      mathtools::meshgrid<double>(xmesh, xmesh);

  mathtools::write_vector(centers[0],"centers_x.txt");
  mathtools::write_vector(centers[1],"centers_y.txt");
  local_lagrange::LocalLagrangeConstructor llc;
  llc.setCenters(centers[0], centers[1]);
  llc.assembleTree();
  llc.setNum_local_centers(200);
 

// for (size_t iter = 0; iter < num_centers; iter++) {
  size_t iter = 1341;
    std::cout << "The iteration is " << iter << std::endl;
    local_lagrange::LocalLagrange llf =
        llc.generateLocalLagrangeFunction(iter);
    arma::vec coefs = llf.coefficients();
    EXPECT_NEAR(0, accu(coefs.subvec(0, 199)), 1e-11);
    double x_eval = 0;
    double y_eval = 0;
    arma::vec coef_tps = coefs.subvec(0, 199);
    std::vector<unsigned int> local_indices = llf.indices();
    std::string index_file = "indicies_" + std::to_string(iter)+".txt";
    std::string coefs_file = "coefs_" + std::to_string(iter)+".txt";
    mathtools::write_vector(local_indices,index_file);
    bool save_status = coefs.save(coefs_file,arma::raw_ascii);
    EXPECT_TRUE(save_status);
    for (size_t eval_iter = 0; eval_iter < 200; eval_iter++) {
      x_eval += coef_tps(eval_iter) * centers[0][local_indices[eval_iter]];
      y_eval += coef_tps(eval_iter) * centers[1][local_indices[eval_iter]];
    }

}

int main(int argc, char **argv ) {
  ::testing::InitGoogleTest(&argc, argv);
  int return_value = RUN_ALL_TESTS();

  return return_value;
}
