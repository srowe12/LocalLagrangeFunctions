#include "../local_lagrange.h"
#include "math_tools.h"
#include <gtest/gtest.h>
#include <stdio.h>
#include <utility>
#include <string>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <chrono>
//#include <thread>
#include <future>
void BuildLocalLagrange(local_lagrange::LocalLagrangeConstructor& llc, size_t iter) {
//   std::cout << "The iteration is" << iter << std::endl;
   local_lagrange::LocalLagrange llf = llc.generateLocalLagrangeFunction(iter);
}
int main() {
   //Purpose of this app is to time how long it takes to generate a full set of LLF's.


   //Build some centers; say 2601 of them.
  std::cout << "Beginning run!" << std::endl;
  auto start = std::chrono::steady_clock::now();
  size_t num_points = 50; 
  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  std::array<std::vector<double>, 2> centers =
      mathtools::meshgrid<double>(xmesh, xmesh);

  //TODO: Constructor is kinda dumb...
  //Set the centers into the Local Lagrange Constructor...should rename it assembler...
  local_lagrange::LocalLagrangeConstructor llc;
  llc.setCenters(centers[0], centers[1]);
  //Assemble the R-Tree to bin the centers based on squares, not circles. Close enough.
  llc.assembleTree();
  //Specify how many local centers we want to use.
  size_t num_local_centers = 200;
  llc.setNum_local_centers(num_local_centers);
 
 size_t num_centers = centers[0].size();
 
 #pragma omp parallel for num_threads(6)
    for (size_t iter =0; iter< num_centers; ++iter) {
 //     std::cout << "The iteration is" << iter << std::endl;
      local_lagrange::LocalLagrange  llf = llc.generateLocalLagrangeFunction(iter);
    }
/*
 std::vector<std::future<void>> futures;
 for (size_t i = 0; i < num_centers; ++i) {
   futures.push_back(std::async(BuildLocalLagrange,std::ref(llc),i));
 }
 
 for (auto &e : futures) {
   e.get();
 }
 */
/*
 for (size_t iter = 0; iter < num_centers; iter++) {
    std::cout << "The iteration is " << iter << std::endl;
    local_lagrange::LocalLagrange llf =
        llc.generateLocalLagrangeFunction(iter);
    arma::vec coefs = llf.coefficients();
    std::string index_file = "indices_" + std::to_string(iter)+".txt";
    std::string coefs_file = "coefs_" + std::to_string(iter)+".txt";
    mathtools::write_vector(local_indices,index_file);
    bool save_status = coefs.save(coefs_file,arma::raw_ascii);
 }    
 */
 
 std::cout << "Run complete!" << std::endl;
 auto end = std::chrono::steady_clock::now();
 auto diff = end-start;
 std::cout << std::chrono::duration<double,std::deci> (diff).count() << "deciseconds" << std::endl; 

 return 0;
}
