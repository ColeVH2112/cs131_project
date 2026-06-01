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
    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    if N < 2:
        return {i: [] for i in range(N)}

    k_eff = min(k, N - 1)

    graph: dict[int, list[tuple[int, float]]] = {i: [] for i in range(N)}

    # Memory-bounded k-NN. Materialising the full (N, N, 3) difference tensor is
    # O(N^2) memory (~44 GB at N=43k) and OOM-kills the kernel on the 20k-40k-point
    # filtered clouds. Instead we compute squared distances one ROW BLOCK at a time
    # via |a-b|^2 = |a|^2 + |b|^2 - 2 a·b, so peak memory is O(BLOCK * N), not
    # O(N^2). Still hand-rolled (no scipy.cKDTree) to stay inside the from-scratch
    # boundary; argpartition on squared distance picks the same neighbours as on
    # distance, so the resulting graph is identical to the naive version.
    sq = np.einsum("ij,ij->i", points, points)         # |x|^2 per point, (N,)
    BLOCK = 512
    for start in range(0, N, BLOCK):
        end = min(start + BLOCK, N)
        block = points[start:end]                      # (B, 3)
        d2 = sq[start:end, None] + sq[None, :] - 2.0 * (block @ points.T)  # (B, N)
        np.maximum(d2, 0.0, out=d2)                     # floor tiny negatives
        for bi in range(end - start):
            i = start + bi
            row = d2[bi]
            row[i] = np.inf                             # exclude self-edge
            nearest = np.argpartition(row, k_eff)[:k_eff]
            for j in nearest:
                d = float(np.sqrt(row[int(j)]))
                if max_edge_length is not None and d > max_edge_length:
                    continue
                graph[i].append((int(j), d))

    # Symmetrise: if i picked j, ensure j has i. Avoid duplicate edges.
    existing = {(i, j) for i in graph for j, _ in graph[i]}
    for i in list(graph.keys()):
        for j, d in list(graph[i]):
            if (j, i) not in existing:
                graph[j].append((i, d))
                existing.add((j, i))

    return graph
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
    import heapq

    distances = np.full(n_nodes, np.inf, dtype=np.float64)
    distances[root] = 0.0
    parent_of: dict[int, int] = {}
    visited = np.zeros(n_nodes, dtype=bool)

    # (distance, node) min-heap.
    heap: list[tuple[float, int]] = [(0.0, root)]

    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        if d > distances[u]:
            continue
        for v, w in graph.get(u, []):
            new_d = d + w
            if new_d < distances[v]:
                distances[v] = new_d
                parent_of[v] = u
                heapq.heappush(heap, (new_d, v))

    return distances, parent_of
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
    parent_of = dict(parent_of)   # don't mutate the caller's dict
    nodes = np.asarray(nodes, dtype=np.float64)

    # Iterate until no spur is short enough to remove. Each pass:
    # 1. Find leaves (nodes that are no one's parent).
    # 2. For each leaf, walk up summing edge lengths until either the root
    #    is reached or a branching node (a parent with multiple children)
    #    is reached. If total length < min_length, delete every node along
    #    that walk from the parent map.
    while True:
        # Children-of map for the current state.
        children_of: dict[int, list[int]] = {}
        for c, p in parent_of.items():
            children_of.setdefault(p, []).append(c)

        all_children = set(parent_of.keys())
        all_parents = set(parent_of.values())
        leaves = all_children - all_parents

        removed_any = False
        for leaf in list(leaves):
            if leaf not in parent_of:
                continue
            walk = []
            total_len = 0.0
            current = leaf
            while current in parent_of:
                parent = parent_of[current]
                seg = float(np.linalg.norm(nodes[current] - nodes[parent]))
                total_len += seg
                walk.append(current)
                # Stop at a branching ancestor — don't delete the branching
                # node itself, only the spur dangling off it.
                if len(children_of.get(parent, [])) > 1:
                    break
                current = parent

            if total_len < min_length:
                for n in walk:
                    parent_of.pop(n, None)
                removed_any = True

        if not removed_any:
            break

    return parent_of
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
    parent_of = dict(parent_of)
    nodes = np.asarray(nodes, dtype=np.float64)
    cos_tol = float(np.cos(np.deg2rad(angle_tolerance_deg)))

    # Iteratively bypass any node that has exactly one child AND whose
    # incoming/outgoing edge directions agree within angle_tolerance_deg.
    while True:
        children_of: dict[int, list[int]] = {}
        for c, p in parent_of.items():
            children_of.setdefault(p, []).append(c)

        bypassed_any = False
        for node, kids in list(children_of.items()):
            if node not in parent_of:
                continue   # root — has no parent edge to merge across
            if len(kids) != 1:
                continue   # branching or leaf node — preserve
            child = kids[0]
            parent = parent_of[node]

            in_dir  = nodes[node]  - nodes[parent]
            out_dir = nodes[child] - nodes[node]
            in_n  = float(np.linalg.norm(in_dir))
            out_n = float(np.linalg.norm(out_dir))
            if in_n < 1e-9 or out_n < 1e-9:
                continue

            cos_angle = float((in_dir / in_n) @ (out_dir / out_n))
            if cos_angle >= cos_tol:
                # Bypass `node`: re-parent the child onto node's parent.
                parent_of[child] = parent
                parent_of.pop(node, None)
                bypassed_any = True

        if not bypassed_any:
            break

    # Compact the node set: only nodes that still appear as a key or as a
    # value in parent_of are kept. Reindex contiguously starting at 0.
    used = set(parent_of.keys()) | set(parent_of.values())
    if not used:
        return np.empty((0, 3), dtype=np.float64), {}
    used_sorted = sorted(used)
    old_to_new = {old: new for new, old in enumerate(used_sorted)}

    kept_nodes = nodes[used_sorted]
    new_parent_of = {old_to_new[c]: old_to_new[p] for c, p in parent_of.items()}
    return kept_nodes, new_parent_of
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
