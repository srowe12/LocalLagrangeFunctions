#include <armadillo>

// Code inspired and leveraged from:
// https://www.geeksforgeeks.org/k-dimensional-tree/
template <size_t N> struct Node {
  arma::rowvec::fixed<N> point;
  Node<N> *left;
  Node<N> *right;
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

// Searches a point in the KD tree the parameter depth is used to determine
// current axis
template <size_t N>
bool searchRec(std::shared_ptr<Node<N>> root,
               const arma::rowvec::fixed<N> &point, unsigned depth = 0) {
  // Base cases
  if (root == nullptr) {
    return false;
  }
  if (point == root->point) {
    return true;
  }

  unsigned current_depth = depth % N;

  if (point(current_depth) < root->point(current_depth)) {
    return searchRec(root->left, point, depth + 1);
  }

  return searchRec(root->right, point, depth + 1);
}

template <size_t N>
bool search(std::shared_ptr<Node<N>> root,
            const arma::rowvec::fixed<N> &point) {
  return searchRec(root, point, 0);
}

arma::mat SortOnColumnIndex(const arma::mat &points, const unsigned dimension) {
  // Slice down the column
  arma::uvec indices = arma::sort_index(points.col(dimension));

  // Sort

  return std::move(points(indices, arma::span::all()));
}

template <size_t N>
std::shared_ptr<Node<N>> BuildTree(const arma::mat &points, depth = 0) {
  unsigned axis = depth % N;

  arma::mat sorted_points = SortOnColumnIndex(points, axis);

  auto median_point = sorted_points.rows / 2; ///@todo srowe: Not really right
  // https://en.wikipedia.org/wiki/K-d_tree
  return std::make_shared<Node>{
      points.row(median_point),
      BuildTree(points.rows(0, median_point), depth + 1),
      BuildTree(points.rows(median_point + 1, sorted_points.rows), depth + 1)};
}
