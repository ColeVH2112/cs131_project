"""Stage 5a — trunk axis fit via RANSAC (FROM SCRATCH).

The trunk is the single thickest near-vertical structure in the filtered point
cloud. We fit it as a 3D line via RANSAC over the cleaned cloud:

    for N iterations:
        sample 2 points → candidate line (anchor + direction)
        count inliers — points within `inlier_thresh` of the line
        track best inlier set

    refit the best line via least-squares (PCA) over its inlier subset
    pick a "root" = inlier with the most negative axis coord (trunk base)

This is adapted from Lec. 3 (Hough-line RANSAC) — same fit-by-random-sample
idea, but in 3D and with a line model rather than 2D Hough accumulators.

Outputs feed `branch_graph.py`, which uses the trunk root as the Dijkstra
source node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrunkFit:
    """A fitted trunk: line parameters + the points it claims as inliers."""
    root: np.ndarray             # (3,) — point on the line nearest the trunk base
    direction: np.ndarray        # (3,) — unit vector pointing up the trunk
    inlier_mask: np.ndarray      # (N,) bool over the input cloud
    inlier_distance_mean: float  # mean perpendicular distance of inliers to the line


# ---------------------------------------------------------------------------
# Helpers (also from-scratch credit since they live inside the trunk fit)
# ---------------------------------------------------------------------------

def point_to_line_distance(
    points: np.ndarray, anchor: np.ndarray, direction: np.ndarray,
) -> np.ndarray:
    """Perpendicular distance from each point to an infinite 3D line.

    Line is parameterised as L(t) = anchor + t · direction (direction need
    not be unit; we normalise internally).

    Args:
        points:    (N, 3) array of points.
        anchor:    (3,) point on the line.
        direction: (3,) direction vector.

    Returns:
        (N,) per-point perpendicular distance.
    """
    ### YOUR CODE HERE
    points = np.asarray(points, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)
    v = points - np.asarray(anchor, dtype=np.float64)[None, :]
    # Projection of v onto d, then subtract to get the perpendicular component.
    proj = (v @ d)[:, None] * d[None, :]
    perp = v - proj
    return np.linalg.norm(perp, axis=1)
    ### END YOUR CODE


def fit_line_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares 3D line fit via PCA.

    The best-fit line passes through the centroid of the points and is
    aligned with the first principal direction (the eigenvector of the
    covariance with the largest eigenvalue).

    Args:
        points: (N, 3) array — should be the RANSAC inlier set.

    Returns:
        (anchor, direction):
            anchor:    (3,) centroid of `points`.
            direction: (3,) unit vector of the principal axis.
    """
    ### YOUR CODE HERE
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    # SVD on the centred matrix; rows of Vt are principal directions, sorted
    # by descending singular value. The first row is the line's axis.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return centroid, direction
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# The RANSAC trunk fitter itself.
# ---------------------------------------------------------------------------

def ransac_trunk_axis(
    points: np.ndarray,
    num_iter: int = 1000,
    inlier_thresh: float = 0.05,
    vertical_prior: bool = True,
    vertical_axis: int = 2,
    vertical_cos_min: float = 0.7,
    rng: np.random.Generator | None = None,
) -> TrunkFit:
    """RANSAC fit of a 3D line for the trunk axis.

    Standard RANSAC loop, with one tree-specific tweak: trees are mostly
    vertical, so we optionally reject any sample whose direction is too far
    off the world up axis. This is controlled by `vertical_prior`,
    `vertical_axis` (the world axis that points up — depends on which way
    the COLMAP frame ended up), and `vertical_cos_min` (require
    |dot(direction, up_axis)| ≥ this value).

    Algorithm:

        for _ in range(num_iter):
            sample 2 distinct points → candidate line
            if vertical_prior and not vertical-enough: continue
            inliers = points within inlier_thresh of the line
            if more inliers than current best: update best

        anchor, direction = fit_line_pca(best_inliers)
        root = inlier with smallest signed projection on `direction`

    Args:
        points: (N, 3) filtered cloud.
        num_iter: number of RANSAC iterations.
        inlier_thresh: perpendicular-distance threshold in metres.
        vertical_prior: enforce a near-vertical line.
        vertical_axis: which of {0,1,2} is the world up axis.
        vertical_cos_min: min |cos(angle to up axis)| if vertical_prior is on.
        rng: optional numpy Generator for reproducibility.

    Returns:
        TrunkFit with the recovered line, inlier mask, and mean inlier distance.
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to fit a trunk line.")

    if rng is None:
        rng = np.random.default_rng(131)

    ### YOUR CODE HERE
    points = np.asarray(points, dtype=np.float64)
    N = len(points)

    up_axis = np.zeros(3, dtype=np.float64)
    up_axis[vertical_axis] = 1.0

    best_inlier_count = -1
    best_inlier_mask = None

    for _ in range(num_iter):
        # Sample two distinct points → candidate line.
        idx = rng.choice(N, size=2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        d_vec = p2 - p1
        d_norm = np.linalg.norm(d_vec)
        if d_norm < 1e-9:
            continue
        d_unit = d_vec / d_norm

        # Tree-specific tweak: skip candidates that don't roughly point up.
        if vertical_prior and abs(float(d_unit @ up_axis)) < vertical_cos_min:
            continue

        dists = point_to_line_distance(points, p1, d_unit)
        inliers = dists < inlier_thresh
        count = int(inliers.sum())
        if count > best_inlier_count:
            best_inlier_count = count
            best_inlier_mask = inliers

    if best_inlier_mask is None or best_inlier_count < 2:
        raise RuntimeError(
            "RANSAC trunk fit found no usable hypothesis. Try lowering "
            "vertical_cos_min, raising inlier_thresh, or running more iterations."
        )

    # Refit the line with PCA over the inlier set for a tighter estimate.
    anchor, direction = fit_line_pca(points[best_inlier_mask])

    # Re-pick the inlier set under the refined line, since PCA may have
    # moved the axis enough that a couple of marginal points cross the
    # threshold either way.
    final_dists = point_to_line_distance(points, anchor, direction)
    inlier_mask = final_dists < inlier_thresh

    # Root = inlier with the smallest signed projection on `direction`
    # (i.e. the trunk base, the negative end of the axis).
    inlier_pts = points[inlier_mask]
    projs = (inlier_pts - anchor) @ direction
    root = inlier_pts[int(np.argmin(projs))]

    inlier_distance_mean = (
        float(final_dists[inlier_mask].mean()) if inlier_mask.any() else 0.0
    )

    return TrunkFit(
        root=root,
        direction=direction,
        inlier_mask=inlier_mask,
        inlier_distance_mean=inlier_distance_mean,
    )
    ### END YOUR CODE


def trunk_height(fit: TrunkFit, points: np.ndarray) -> float:
    """Compute trunk height — extent of inliers projected onto the trunk axis."""
    inliers = points[fit.inlier_mask]
    if len(inliers) == 0:
        return 0.0
    projs = (inliers - fit.root) @ fit.direction
    return float(projs.max() - projs.min())
