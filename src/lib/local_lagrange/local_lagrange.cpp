#include "local_lagrange.h"
#include <math.h>
#include <armadillo>
#include <boost/math/quaternion.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>

#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace local_lagrange{

typedef bg::model::point<double, 2, bg::cs::cartesian> point;
typedef std::pair<point, unsigned> value;


LocalLagrange generateLocalLagrangeFunction(unsigned int index){
   //Should the Local Lagrange do this?
   

    LocalLagrange llf(index);
     
    

}

void LocalLagrangeConstructor::assembleTree(){
   bgi::rtree<value, bgi::quadratic<16> > rt;   
   std::vector<value> points;
   for (int iter =0; iter<centers_x_.size(); iter++) {
       point mypoint(centers_x_[iter],centers_y_[iter]);
       value myvalue(mypoint,iter);
       points.push_back(myvalue);
   }
   rt.insert(points.begin(),points.end() );
    

}
void LocalLagrangeConstructor::getNearestNeighbors(unsigned int index){

}
} //namespace local_lagrange
