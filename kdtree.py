from typing import Tuple
import numpy as np

class Node:
    """A single node used in the construction of a k-dimensional (k-d) tree
    data structure for a 3-dimensional (3D) point cloud.

    Attributes:
        point (tuple[float, float, float]): The (x, y, z) coordinates of the point.
        axis (int): Splitting axis used at this node (0 = x, 1 = y, 2 = z).
        left (int | None): Index of the left child node in the k-d tree array,
            or None if this node has no left child.
        right (int | None): Index of the right child node in the k-d tree array,
            or None if this node has no right child.
    """

    __slots__ = ("point", "axis", "left", "right")

    def __init__(
            self,
            point: Tuple[float, float, float],
            axis: int,
            left: int | None,
            right: int | None
    ):
        """Initialize a KD-tree node."""

        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

def build(Q: np.ndarray) -> Tuple[Node, ...]:
    """Construct a pre-ordered KD-tree from a point cloud.

    Splits at the median along alternating axes (x=0, y=1, z=2) recursively.
    Returns the tree as a tuple of nodes for immutability.

    Args:
        P (np.ndarray): Array of shape (M, 3) containing M points.

    Returns:
        tuple: Pre-ordered k-d tree as a tuple of Node objects.
    """

    nodes: list[Node] = [None] * Q.shape[0]
    idx = 0

    def _build(points: np.ndarray, depth: int) -> int:
        """Recursive helper to fill nodes list in pre-order sequence.

        Args:
            points (np.ndarray): Subset of points to build tree from.
            depth (int): Current depth in tree (to determine axis).

        Returns:
            int: Index of this node in the nodes list.
        """
        nonlocal idx

        # Empty child
        if points.shape[0] == 0:
            return None
        
        # Cycle through splitting axes for each layer starting with x (0)
        axis = depth % 3

        # Sort points along current axis and choose median
        sorted_idx = np.argsort(points[:, axis])
        median_idx = len(points) // 2
        median_point = points[sorted_idx[median_idx]]

        # Reserve index in nodes list for this node
        node_idx = idx
        idx += 1

        # Recursively build the left and right subtrees
        left_child = _build(points[sorted_idx[:median_idx]], depth + 1)
        right_child = _build(points[sorted_idx[1 + median_idx:]], depth + 1)

        # Fill node in pre-allocated list
        nodes[node_idx] = Node(tuple(median_point), axis, left_child, right_child)

        # Return the tree index of the current node
        return node_idx
    
    # Recursively construct the tree starting from the root
    _build(Q, depth=0)

    # Return the tree as a tuple
    return tuple(nodes)

def nn_search(
    tree: Tuple[Node, ...],
    P: np.ndarray,
    down_eval: bool=True
) -> np.ndarray:
    """Find nearest neighbors in a k-d tree for each point in a point cloud.

    Args:
        P (np.ndarray): Array of shape (N, 3) containing N points to query.
        tree (tuple[Node, ...]): KD-tree represented as a tuple of Node objects,
            as returned by `build_kd_tree`.

    Returns:
        tuple: `Q_nearest` and `[n_euclidean, n_split]` outlined below.

        **Q_nearest** :  *np.ndarray*<br>
        Array of shape (N, 3) containing the nearest neighbors for each point in P.

        **[n_euclidean, n_split]** : *list*<br>
        List containing counts for the total number of euclidean (`n_euclidean`) and
        split (`n_split`) distance calculations.
    """

    Q_nearest = np.empty_like(P)

    n_euclideans = [0]*P.shape[0]
    n_splits = [0]*P.shape[0]

    def _nn_search(
        point:np.ndarray,
        node_idx: int,
        best: Tuple[int, float]
    ) -> Tuple[int, float]:
        """Recursive helper to find the nearest neighbor of a single point.

        Args:
            point (np.ndarray): Query point of shape (3,).
            node_idx (int): Index of the current node in the tree.
            best (Tuple[int, float]): Tuple containing the index of the best node
                found so far and its squared distance.

        Returns:            
            best (Tuple[int, float]): Updated best node index and squared distance.
        """
        nonlocal n_euclidean, n_split, down_eval

        # Leaf node
        if node_idx is None:
            return best
        
        # Current node
        node = tree[node_idx]
        node_point = np.array(node.point)

        def _euclidean_distance(best):
            """Helper function to calculate the current euclidean distance and update
            the best point.
            """
            nonlocal point, node_point, n_euclidean

            # Calculate the current euclidean distance
            dist_sq = np.sum((point - node_point) ** 2)
            n_euclidean += 1

            # Update best if current node is closer
            if dist_sq < best[1]:
                best = (node_idx, dist_sq)

            return best

        # Compute the euclidean distance and update best on the down traversal
        if down_eval:
            best = _euclidean_distance(best)

        # Check on which side of the current nodes splitting axis the point lies
        if point[node.axis] < node_point[node.axis]:
            near, far = node.left, node.right           # near (or same) side
        else:
            near, far = node.right, node.left           # far (or opposite) side

        # Explore the near sub-tree
        best = _nn_search(point, near, best)

        # Compute the euclidean distance and update best on the up traversal
        if not down_eval:
            best = _euclidean_distance(best)

        # Determine if we need to explore the far sub-tree
        split = (point[node.axis] - node_point[node.axis]) ** 2     # split distance
        n_split += 1
        if split < best[1]:
            best = _nn_search(point, far, best)

        return best

    # Find the nearest point for each point in P
    for i in range(P.shape[0]):

        # Counters to track the number of euclidean and split distance calculations
        n_euclidean = 0
        n_split = 0

        # Fine the nearest neighbor in Q for the current point in P
        best_idx, _ = _nn_search(P[i], 0, (None, float("inf")))

        # Add the nearest point to the new point cloud
        Q_nearest[i] = np.array(tree[best_idx].point)

        # Add the total euclidean and split distance calculations counts to the output
        n_euclideans[i] = n_euclidean
        n_splits[i] = n_split

    return Q_nearest, [n_euclideans, n_splits]
