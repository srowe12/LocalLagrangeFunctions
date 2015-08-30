#include <stdio.h>
#include <math.h>
#include <vector>
#include <armadillo>

namespace local_lagrange {
class LocalLagrange{
public:

explicit LocalLagrange(unsigned int index) : index_(index) {}
LocalLagrange(unsigned int index, std::vector<double> coefs, std::vector<double> indices) :index_(index), indices_(indices), coefficients_(coefs) {}


arma::mat assemble_interpolation_matrix();

private:
unsigned int index_;
std::vector<double> indices_;
std::vector<double> coefficients_;
};

class LocalLagrangeConstructor{
public: 

LocalLagrangeConstructor() : num_centers_(0), scale_factor_(1), mesh_norm_(0) {updateBallRadius();}

LocalLagrange generateLocalLagrangeFunction(unsigned int index);
unsigned int num_centers() const {return num_centers_;}
std::vector<double> centers_x() const {return centers_x_;}
std::vector<double> centers_y() const {return centers_y_;}


void assembleTree();

void getNearestNeighbors(unsigned int index);

void setScale_factor(double scale_factor) {scale_factor_= scale_factor; updateBallRadius();}
void setMesh_norm(double mesh_norm) {mesh_norm_ = mesh_norm; updateBallRadius();}
void setCenters(std::vector<double> centers_x, std::vector<double> centers_y) {
    //Assumes size of centers_x and centers_y are the same
    centers_x_ = centers_x;
    centers_y_ = centers_y;
    num_centers_ = centers_x_.size();    
}
private:

void updateBallRadius() { ball_radius_ = scale_factor_*mesh_norm_*abs(log(mesh_norm_));}

unsigned int num_centers_;
double scale_factor_;  //We use ball_radius = scale_factor*mesh_norm*abs(log(mesh_norm));
double mesh_norm_;
double ball_radius_;

std::vector<double> centers_x_;
std::vector<double> centers_y_; //Assumes 2D structure.
};

} // namespace local_lagrange
