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
  arma::uvec indices = arma::sort_index(points.col(dimension));

  // Sort

  return points.rows(indices);
}

template <size_t N>
std::shared_ptr<Node<N>> BuildTree(const arma::mat &points, const unsigned int depth = 0) {
  unsigned axis = depth % N;
  std::cout << "Building Tree with axis" << axis << std::endl;
  arma::mat sorted_points = SortOnColumnIndex(points, axis);

  auto median_point = sorted_points.n_rows / 2; ///@todo srowe: Not really right
  // https://en.wikipedia.org/wiki/K-d_tree
  arma::rowvec median_row = points.row(median_point);

  std::cout << "The median point is" << median_point << std::endl;
  if (sorted_points.n_rows < 2) {
    arma::rowvec point = sorted_points.row(1);
    return std::make_shared<Node<N>>(point);
  }
  return std::make_shared<Node<N>>(
      median_row, BuildTree<N>(points.rows(0, median_point - 1), depth + 1),
      BuildTree<N>(points.rows(median_point + 1, sorted_points.n_rows),
                   depth + 1));
}
