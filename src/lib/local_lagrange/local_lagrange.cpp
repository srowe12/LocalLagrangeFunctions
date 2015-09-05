#include "local_lagrange.h"
#include <math.h>
#include <armadillo>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>

#include <boost/geometry/index/rtree.hpp>


#include <stdio.h> //Debugging purposes, don't judge me.

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
//   bgi::rtree<value, bgi::quadratic<16> > rt;   
   std::vector<value> points;
   for (size_t iter =0; iter<centers_x_.size(); iter++) {
       point mypoint(centers_x_[iter],centers_y_[iter]);
       value myvalue(mypoint,iter);
       points.push_back(myvalue);
   }
   rt_.insert(points.begin(),points.end() );
    

}
std::vector<unsigned>  LocalLagrangeConstructor::getNearestNeighbors(unsigned int index){
     //Wrap values into a single point, then value pair. Pass into rt for querying.
     point center(centers_x_[index],centers_y_[index]);
     value center_value(center,index);
     std::vector<value> neighbors;
     //TODO: Update the number 2 to actually be representative.
     rt_.query(bgi::nearest(center,2), std::back_inserter(neighbors));
     
    std::vector<unsigned> indices; 
    for (auto it = neighbors.begin(); it!= neighbors.end(); ++it){
         indices.push_back(std::get<1>(*it));
     }
    return indices; 
}
} //namespace local_lagrange
