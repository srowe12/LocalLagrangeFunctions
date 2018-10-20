#include <armadillo>

// Code inspired and leveraged from:
// https://www.geeksforgeeks.org/k-dimensional-tree/
template <size_t N> struct Node {

  explicit Node(arma::rowvec &point)
      : point(point), left(nullptr), right(nullptr) {}
  Node(arma::rowvec &point, std::shared_ptr<Node<N>> l,
       std::shared_ptr<Node<N>> r)
      : point(point), left(l), right(r) {}
  arma::rowvec point;
  std::shared_ptr<Node<N>> left;
  std::shared_ptr<Node<N>> right;
};

template <size_t N>
std::shared_ptr<Node<N>> newNode(const arma::rowvec::fixed<N> &point) {
  Node<N> temp = std::make_shared<Node<N>>();

  temp->point = point;
  temp->left = nullptr;
  temp->right = nullptr;
  return std::make_shared(temp);
}

template <size_t N>
// Inserts a new node and returns root of the modified tree
// The parameter depth is used to decide the axis of comparison, hence
// the depth % N
std::shared_ptr<Node<N>> insertRec(std::shared_ptr<Node<N>> root,
                                   const arma::rowvec::fixed<N> &point,
                                   unsigned depth) {

  // Tree is empty?
  if (root == nullptr) {
    return newNode(point);
  }

  unsigned current_depth = depth % N;

  if (point(current_depth) < (root->point(current_depth))) {
    root->left = insertRec(root->left, point, depth + 1);
  } else {
    root->right = insertRec(root->right, point, depth + 1);
  }

  return root;
}

template <size_t N>
std::shared_ptr<Node<N>> insert(std::shared_ptr<Node<N>> root,
                                const arma::rowvec::fixed<N> &point) {
  return insertRec(root, point, 0);
}

template <size_t N>
bool ComparePoints(const arma::rowvec& p1, const arma::rowvec& p2) {
	///@todo srowe; Use other stuff already written
	double diff  = 0.0;
	for (size_t i = 0; i < N; ++i) {
		diff += (p1[i]-p2[i])*(p1[i]-p2[i]);
	}
	return diff < 1e-13;
}

// Searches a point in the KD tree the parameter depth is used to determine
// current axis
template <size_t N>
bool searchRec(std::shared_ptr<Node<N>> root,
               const arma::rowvec &point, unsigned depth = 0) {
  // Base cases
  if (root == nullptr) {
    return false;
  }
  arma::rowvec root_point = root->point;
  if (ComparePoints<N>(point, root_point)) {
    return true;
  }

  unsigned current_depth = depth % N;

  if (point(current_depth) < root->point(current_depth)) {
    return searchRec<N>(root->left, point, depth + 1);
  }

  return searchRec<N>(root->right, point, depth + 1);
}

template <size_t N>
bool search(std::shared_ptr<Node<N>> root,
            const arma::rowvec &point) {
  return searchRec<N>(root, point, 0);
}

inline arma::mat SortOnColumnIndex(const arma::mat &points, const unsigned dimension) {
  // Slice down the column
  std::cout << "Sorting" << std::endl;
  arma::uvec indices = arma::sort_index(points.col(dimension));

  // Sort

  return points.rows(indices);
}

template <size_t N>
std::shared_ptr<Node<N>> BuildTree(const arma::mat &points, const unsigned int depth = 0) {
  unsigned axis = depth % N;
  std::cout << "Building Tree with axis " << axis << std::endl;
  arma::mat sorted_points = SortOnColumnIndex(points, axis);

  auto median_point = sorted_points.n_rows / 2; ///@todo srowe: Not really right
  // https://en.wikipedia.org/wiki/K-d_tree
  arma::rowvec median_row = sorted_points.row(median_point);

  const size_t sorted_rows = sorted_points.n_rows;
  // Make the left one
  std::shared_ptr<Node<N>> left = std::shared_ptr<Node<N>>(nullptr);
sorted_points.print("The available points are");
  int left_length = median_point;
  
  if (left_length > 1) {
	  arma::mat left_points = sorted_points.rows(0, median_point - 1);
    left = BuildTree<N>(left_points, depth + 1);
  }
  else {
	  // Leaf point
	  std::cout << " left_length = " << left_length << " so adding row(0)" << std::endl;
	  arma::rowvec left_point = sorted_points.row(0);
	  left = std::make_shared<Node<N>>(left_point, nullptr, nullptr);
  }


  int right_length = sorted_rows - median_point;
  std::shared_ptr<Node<N>> right = std::shared_ptr<Node<N>>(nullptr);

  if (right_length > 1) {
	    arma::mat right_points =
      sorted_points.rows(median_point + 1, sorted_rows - 1);
	  std::cout << " Going down the right tree now!" << std::endl;
    right = BuildTree<N>(right_points, depth + 1);
  }
  /*
  else {
	  // We have a final node
	  arma::rowvec right_point = sorted_points.row(median_point+1);
	  right = std::make_shared<Node<N>>(right_point, nullptr, nullptr);
  }
  */
std::cout << "The left and right lengths are " << left_length << " , " << right_length;
std::cout << "With median point " << median_point << " and size of sorted_points " << sorted_points.n_rows << std::endl;
  median_row.print("Adding this point!");
  return std::make_shared<Node<N>>(median_row, left, right);
}
