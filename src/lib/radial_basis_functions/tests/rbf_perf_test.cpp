#include <rbf/gaussian.h>
#include <rbf/interpolate.h>

using namespace rbf; 

arma::vec sampleFunction(const arma::mat& sample_points) {
    const arma::vec x = sample_points.col(0);
    const arma::vec y = sample_points.col(1);

    const arma::vec z = arma::sin(2*M_PI*x)%arma::cos(2*M_PI*y);

    return z;
}

void buildRbfInterpolant(const int num_points) {
  std::vector<double> xmesh = mathtools::linspace<double>(0, 1, num_points);
  auto centers = mathtools::meshgrid<double>(xmesh, xmesh);
  const arma::vec data = sampleFunction(centers);

  size_t numcenters = centers.n_rows;
  auto start = std::chrono::steady_clock::now();
  Gaussian<double> gaussian(1.0);
  RadialBasisFunctionInterpolant<Gaussian<double>, 1>(gaussian, centers, data);  
  
  auto end = std::chrono::steady_clock::now();
  std::cout << "Run complete!" << std::endl;
  auto diff = end - start;
  double time_diff = std::chrono::duration<double, std::milli>(diff).count();

  std::cout << time_diff << "milliseconds" << std::endl;

}

int main() {
  std::vector<int> num_points{50,75,100};
  for (const auto num : num_points) {
     buildRbfInterpolant(num);
  }

}
