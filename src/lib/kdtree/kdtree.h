#ifndef LOCAL_LAGRANGE_KDTREE_H
#define LOCAL_LAGRANGE_KDTREE_H

#include <memory>
#include <armadillo>

template <size_t Dimension>
class Node {
public:
    Node(const arma::rowvec::fixed<Dimension>& location_in, std::shared_ptr<Node<Dimension>> left_node, std::shared_ptr<Node<Dimension>> right_node) :
    location(location_in), left_child(left_node), right_child(right_node) {}
	arma::rowvec::fixed<Dimension> location;
	std::shared_ptr<Node<Dimension>> left_child;
	std::shared_ptr<Node<Dimension>> right_child;

	void Print() {
		location.print("Location");
		std::cout << "Left Children " << std::endl;
		if (left_child) {
			left_child->Print();
		}
		std::cout << "Right Children" << std::endl;
		if (right_child) {

			right_child->Print();
		}
	}
	
};

template <size_t Dimension>
std::shared_ptr<Node<Dimension>> BuildTree(arma::mat& point_list, size_t depth=0) {
    
    auto k = point_list.n_rows;

     if (k == 1 ) {
     	arma::rowvec::fixed<Dimension> last_row = point_list.row(0);
         return std::make_shared<Node<Dimension>>(last_row, std::shared_ptr<Node<Dimension>>(nullptr), std::shared_ptr<Node<Dimension>>(nullptr));
     }
    
    auto axis = fmod(depth, Dimension);

    // Sort on axis of point list
    point_list.print("Point list");
    size_t median = k / 2;
    std::cout << "The median is" << median << "the k is " << k << "and the axis is " << axis << std::endl;
    arma::rowvec::fixed<Dimension> median_point = point_list.row(median);
    arma::mat left_points = point_list.rows(0, median-1);
    arma::mat right_points = point_list.rows(median, k-1);
    std::cout << "Got the points, building next tree" << std::endl;
    auto left_child = BuildTree<Dimension>(left_points, depth+1);
    auto right_child = BuildTree<Dimension>(right_points, depth+1);

    return std::make_shared<Node<Dimension>>(median_point, left_child, right_child);

}


#endif //LOCAL_LAGRANGE_KDTREE_H
