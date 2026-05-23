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
    raise NotImplementedError("point_to_line_distance: from-scratch implementation pending.")
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
    raise NotImplementedError("fit_line_pca: from-scratch implementation pending.")
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
    raise NotImplementedError("ransac_trunk_axis: from-scratch implementation pending.")
    ### END YOUR CODE


def trunk_height(fit: TrunkFit, points: np.ndarray) -> float:
    """Compute trunk height — extent of inliers projected onto the trunk axis."""
    inliers = points[fit.inlier_mask]
    if len(inliers) == 0:
        return 0.0
    projs = (inliers - fit.root) @ fit.direction
    return float(projs.max() - projs.min())
