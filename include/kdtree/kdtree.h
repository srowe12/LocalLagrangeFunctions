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
bool ComparePoints(const arma::rowvec &p1, const arma::rowvec &p2) {
  ///@todo srowe; Use other stuff already written
  double diff = 0.0;
  for (size_t i = 0; i < N; ++i) {
    diff += (p1[i] - p2[i]) * (p1[i] - p2[i]);
  }
  return diff < 1e-13;
}

// Searches a point in the KD tree the parameter depth is used to determine
// current axis
template <size_t N>
bool searchRec(std::shared_ptr<Node<N>> root, const arma::rowvec &point,
               unsigned depth = 0) {
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
bool search(std::shared_ptr<Node<N>> root, const arma::rowvec &point) {
  return searchRec<N>(root, point, 0);
}

inline arma::mat SortOnColumnIndex(const arma::mat &points,
                                   const unsigned dimension) {
  // Slice down the column
  std::cout << "Sorting" << std::endl;
  arma::uvec indices = arma::sort_index(points.col(dimension));

  // Sort

  return points.rows(indices);
}

template <size_t N>
std::shared_ptr<Node<N>> BuildTree(const arma::mat &points,
                                   const unsigned int depth = 0) {
  unsigned axis = depth % N;
  arma::mat sorted_points = SortOnColumnIndex(points, axis);

  auto median_point = sorted_points.n_rows / 2; ///@todo srowe: Not really right
  // https://en.wikipedia.org/wiki/K-d_tree
  arma::rowvec median_row = sorted_points.row(median_point);

  const size_t sorted_rows = sorted_points.n_rows;
  // Make the left one
  std::shared_ptr<Node<N>> left = std::shared_ptr<Node<N>>(nullptr);
  int left_length = median_point;

  if (left_length > 1) {
    arma::mat left_points = sorted_points.rows(0, median_point - 1);
    left = BuildTree<N>(left_points, depth + 1);
  } else {
    // Leaf point

    arma::rowvec left_point = sorted_points.row(0);
    left = std::make_shared<Node<N>>(left_point, nullptr, nullptr);
  }

  int right_length = sorted_rows - median_point;
  std::shared_ptr<Node<N>> right = std::shared_ptr<Node<N>>(nullptr);

  if (right_length > 1) {
    arma::mat right_points =
        sorted_points.rows(median_point + 1, sorted_rows - 1);
    right = BuildTree<N>(right_points, depth + 1);
  }
  return std::make_shared<Node<N>>(median_row, left, right);
}

template <size_t N>
std::vector<arma::rowvec> RadiusQuery(std::shared_ptr<Node<N>> tree,
                                      const arma::rowvec &point,
                                      const double radius) {
  std::vector<arma::rowvec> neighbors;

  // Tree is a nullptr, so exit
  if (!tree) {
    return neighbors;
  }
  unsigned int depth = 0;
  RadiusQuery(tree, point, radius, depth, neighbors);

  return neighbors;
}

template <size_t N>
void RadiusQuery(std::shared_ptr<Node<N>> tree, const arma::rowvec &point,
                 const double radius, unsigned int depth,
                 std::vector<arma::rowvec> &neighbors) {
  if (!tree) {
    return;
  }

  const double r2 = radius * radius;
  double distance = 0.0;
  ///@todo srowe: Replace with a better unrolled version in math_tools
  tree->point.print("The current branch point is");
  for (size_t i = 0; i < N; ++i) {
    double dist = tree->point(i) - point(i);
    distance += dist * dist;
  }

  const double one_dim_diff = point(depth) - tree->point(depth);

  const double one_dim_diff_squared = one_dim_diff * one_dim_diff;

  // Is this point sufficiently close to the target point? If so, add it
  if (distance <= r2) {
    neighbors.push_back(tree->point);
  }

  // Which direction of the tree should we choose? We base this off of the
  // current dimension we are comparing

  std::shared_ptr<Node<N>> next_node;
  std::shared_ptr<Node<N>> optional_node;
  if (one_dim_diff > 0) {
    next_node = tree->left;
    optional_node = tree->right;
  } else {
    next_node = tree->right;
    optional_node = tree->left;
  }

  std::cout << "The one dim diff is" << one_dim_diff
            << " and the distance squared is" << distance << std::endl;

  const unsigned int next_depth = (depth + 1) % N;
  RadiusQuery(next_node, point, radius, next_depth, neighbors);

  if (one_dim_diff_squared <= r2) {
    RadiusQuery(optional_node, point, radius, next_depth, neighbors);
  }
}