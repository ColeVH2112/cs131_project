"""Stage 5b — branch graph extraction via Dijkstra (FROM SCRATCH).

Adapted from Livny et al. (2010), but with a much smaller cloud (sparse SfM
instead of dense LiDAR). Pipeline:

    1. Build a weighted undirected graph over the cleaned points: each node
       is connected to its k nearest neighbours, edge weight = Euclidean
       distance.

    2. Run Dijkstra from the trunk root (Lec.-12 graph algorithm). This
       gives, for every point, the shortest path back to the root and a
       parent pointer.

    3. The set of parent pointers IS the shortest-path tree, and that tree
       physically corresponds to the tree's branching structure: real trees
       are tree-shaped graphs (no rejoining), so the geodesic ancestry of any
       branch tip is its actual branch lineage.

    4. Prune spurs — branches with subtree length below a threshold are
       noise (a stray triangulation 5 cm off a real branch); remove them.

    5. Merge collinear segments — split a branch into straight sub-pieces
       via a 3D RANSAC line fit, then drop interior nodes. This is the
       3D analogue of the Hough-line "merge collinear" idea from Lec. 3.

All four stub blocks below are from-scratch credit (look for the standard
`YOUR CODE HERE` / `END YOUR CODE` markers). The NetworkX listed in
requirements.txt is for reference / verification only, not for the
from-scratch implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BranchGraph:
    """The recovered branch skeleton."""
    nodes: np.ndarray                # (N, 3) world coords of skeleton nodes
    edges: list[tuple[int, int]]     # (parent, child) tuples (directed away from root)
    parent_of: dict[int, int]        # child_idx → parent_idx
    distances: np.ndarray            # (N,) geodesic distance from root


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knn_graph(
    points: np.ndarray,
    k: int = 8,
    max_edge_length: float | None = None,
) -> dict[int, list[tuple[int, float]]]:
    """Build a k-nearest-neighbour graph as an adjacency list.

    For each point, connect to its k nearest neighbours. Edges are symmetrised
    (if i lists j as a neighbour, j gets i back even if i wasn't in j's top-k).
    Edges longer than `max_edge_length` are dropped — useful to prevent a
    single bad edge from "stapling" two branches together across a gap.

    Args:
        points: (N, 3) cloud.
        k: number of nearest neighbours per node.
        max_edge_length: drop edges longer than this (same units as points).

    Returns:
        Adjacency list: {node_idx: [(neighbour_idx, weight), ...]} with
        Euclidean-distance weights.
    """
    ### YOUR CODE HERE
    raise NotImplementedError("build_knn_graph: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# Dijkstra shortest-path tree
# ---------------------------------------------------------------------------

def dijkstra_shortest_path_tree(
    graph: dict[int, list[tuple[int, float]]],
    root: int,
    n_nodes: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Dijkstra's algorithm with a binary-heap priority queue.

    Returns the shortest geodesic distance from `root` to every reachable
    node, plus a parent map encoding the resulting shortest-path tree. Nodes
    unreachable from `root` get distance = inf and no parent entry.

    Args:
        graph: adjacency list as built by `build_knn_graph`.
        root: index of the source node (trunk base from `trunk.py`).
        n_nodes: total number of nodes (so we can pre-size the distance array).

    Returns:
        (distances, parent_of):
            distances: (n_nodes,) float — inf where unreachable.
            parent_of: dict child_idx → parent_idx (root not present).
    """
    ### YOUR CODE HERE
    raise NotImplementedError("dijkstra_shortest_path_tree: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# Spur pruning
# ---------------------------------------------------------------------------

def prune_spurs(
    parent_of: dict[int, int],
    nodes: np.ndarray,
    min_length: float = 0.10,
) -> dict[int, int]:
    """Remove tiny side branches ("spurs") from the parent map.

    Walk leaves → root. For each leaf, sum the edge lengths back to the first
    branching node (a node with multiple children). If that subtree length is
    below `min_length`, delete every node in it from the parent map. Repeat
    until no more spurs are removed.

    Args:
        parent_of: child → parent map from `dijkstra_shortest_path_tree`.
        nodes: (N, 3) skeleton positions.
        min_length: cutoff in metres for what counts as "noise".

    Returns:
        New parent_of dict with the spurs removed.
    """
    ### YOUR CODE HERE
    raise NotImplementedError("prune_spurs: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# Collinear segment merging
# ---------------------------------------------------------------------------

def merge_collinear_segments(
    parent_of: dict[int, int],
    nodes: np.ndarray,
    angle_tolerance_deg: float = 8.0,
) -> tuple[np.ndarray, dict[int, int]]:
    """Collapse runs of collinear edges into single edges.

    Walk the tree from the root. Any node with exactly one child whose
    incoming and outgoing edge directions agree within `angle_tolerance_deg`
    is interior to a straight segment — bypass it (re-parent its child onto
    its own parent) and drop it.

    This is the 3D analogue of "collinear merging" from the Lec. 3 Hough
    line ideas referenced in the proposal, and it makes attachment-angle
    measurements stable: without it, the angle of a branch off the trunk
    depends on whichever skeleton node happens to lie closest to the
    attachment point.

    Args:
        parent_of: child → parent map (post-pruning).
        nodes: (N, 3) skeleton positions.
        angle_tolerance_deg: max bend angle for two consecutive edges to be
            considered "collinear".

    Returns:
        (kept_nodes, new_parent_of) where kept_nodes is an (M, 3) subset of
        the original `nodes` and new_parent_of indexes into kept_nodes.
    """
    ### YOUR CODE HERE
    raise NotImplementedError("merge_collinear_segments: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# Glue
# ---------------------------------------------------------------------------

def extract_branch_graph(
    points: np.ndarray,
    trunk_root_xyz: np.ndarray,
    k: int = 8,
    max_edge_length: float | None = 0.30,
    min_spur_length: float = 0.10,
    collinear_tolerance_deg: float = 8.0,
) -> BranchGraph:
    """End-to-end: cloud + trunk root → pruned, merged branch graph.

    Args:
        points: (N, 3) filtered cloud from `filter_cloud.py`.
        trunk_root_xyz: (3,) trunk base from `trunk.py`.
        k: kNN parameter for graph construction.
        max_edge_length: drop edges longer than this when building the graph.
        min_spur_length: minimum surviving subtree length after pruning.
        collinear_tolerance_deg: angle tolerance for collinear merging.

    Returns:
        BranchGraph ready for evaluation / visualisation.
    """
    # Snap the trunk root onto the nearest actual point so we have a real node
    # to use as the Dijkstra source.
    dists_to_root = np.linalg.norm(points - trunk_root_xyz, axis=1)
    root_idx = int(np.argmin(dists_to_root))

    graph = build_knn_graph(points, k=k, max_edge_length=max_edge_length)
    distances, parent_of = dijkstra_shortest_path_tree(graph, root_idx, n_nodes=len(points))
    parent_of = prune_spurs(parent_of, points, min_length=min_spur_length)
    kept_nodes, parent_of = merge_collinear_segments(
        parent_of, points, angle_tolerance_deg=collinear_tolerance_deg,
    )

    edges = [(p, c) for c, p in parent_of.items()]
    # Geodesic distances after pruning/merging — recompute over the new node set
    # for the evaluation stage to use. (Cheap: it's a small tree.)
    new_graph = build_knn_graph(kept_nodes, k=min(k, max(2, len(kept_nodes) - 1)))
    new_dists, _ = dijkstra_shortest_path_tree(new_graph, 0, n_nodes=len(kept_nodes))

    return BranchGraph(
        nodes=kept_nodes, edges=edges,
        parent_of=parent_of, distances=new_dists,
    )
