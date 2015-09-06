#include <gtest/gtest.h>
#include "../local_lagrange.h"
#include <stdio.h>
#include <utility>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>

#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;


typedef bg::model::point<double, 2, bg::cs::cartesian> point;
typedef std::pair<point, unsigned> value;


TEST(MyTest,LocalLagrnageConstructorConstructor){
   local_lagrange::LocalLagrangeConstructor llc;
   EXPECT_EQ(0, llc.num_centers());
   EXPECT_EQ(1, llc.scale_factor());
   EXPECT_EQ(0, llc.mesh_norm() ); 
}

TEST(MyTest, TreeTest){
   std::vector<double> centers_x_{1,2,3};
   std::vector<double> centers_y_{0,1,2};

   bgi::rtree<value, bgi::quadratic<16> > rt;
   std::vector<value> points;
   for (size_t iter =0; iter<centers_x_.size(); iter++) {
       point mypoint(centers_x_[iter],centers_y_[iter]);
       value myvalue(mypoint,iter);
       points.push_back(myvalue);
   }
   rt.insert(points.begin(),points.end() );

   point origin(0,0);
   std::vector<value> results;
   rt.query(bgi::nearest(origin,2), std::back_inserter(results));
   EXPECT_EQ(2, results.size());
   std::pair<point, unsigned>  first_value = results[0];
   point first_point = std::get<0>(first_value);
   EXPECT_EQ(2,first_point.get<0>());
   EXPECT_EQ(1,first_point.get<1>());

   point second_point = std::get<0>(results[1]);
   EXPECT_EQ(1, second_point.get<0>());
   EXPECT_EQ(0, second_point.get<1>());

}

TEST(MyTest, NearestNeighborTest){

   local_lagrange::LocalLagrangeConstructor llc;
   std::vector<double> centers_x{1,2,3};
   std::vector<double> centers_y{0,1,2};
   llc.setCenters(centers_x,centers_y);
   llc.assembleTree();
   std::vector<unsigned> indices = llc.getNearestNeighbors(0);
   EXPECT_EQ(2, indices.size());
   for (auto it = indices.begin(); it != indices.end(); ++it){
      std::cout << *it << std::endl;
   }
}

TEST(MyTest,AssembleInterpolationMatrix){
   std::vector<double> centers_x{1,2,3};
   std::vector<double> centers_y{0,1,2};
   local_lagrange::LocalLagrange llf(0); //Index 0.
   arma::mat interp_matrix = llf.assembleInterpolationMatrix(centers_x,centers_y);
   interp_matrix.print("Interp Matrix:");
}

TEST(MyTest,SolveForCoefficients){
   std::vector<double> centers_x{1,2,3};
   std::vector<double> centers_y{0,1,2};
   unsigned int local_index = 0;
   local_lagrange::LocalLagrange llf(0); //Index 0.
   llf.buildCoefficients(centers_x,centers_y,0);
   arma::vec coefs = llf.coefficients();
   coefs.print("The coefs are:");

   arma::mat interp_matrix = llf.assembleInterpolationMatrix(centers_x,centers_y);
   arma::vec rhs = interp_matrix*coefs;
   EXPECT_NEAR(1,rhs(local_index),1e-15);
   rhs(local_index)=0;
   for (auto it = rhs.begin(); it != rhs.end(); ++it){
       EXPECT_NEAR(0,*it,1e-15);
   }

}
int main(int argc, char** argv) {
   ::testing::InitGoogleTest(&argc, argv);

   
   int return_value = RUN_ALL_TESTS();

   return return_value; 

}
