"""Stage 4b — reprojection-based cloud filtering (FROM SCRATCH).

This is the **core novelty** of the project. The intuition:

    A raw SfM point cloud over a tree contains many ghost points — false
    triangulations from sky texture, lens flares, repeated leaves. Those
    ghosts are not consistent with the segmentation masks across frames:
    if you project a true branch point into every frame, it lands on
    structure pixels (mask == True) almost every time; if you project a
    ghost point, it lands on sky pixels most of the time.

So: for every 3D point, reproject it into every frame, look up the
segmentation mask at the projected pixel, and keep only points whose
hit-rate exceeds a threshold (e.g. 70% of frames where the point is in
view).

This single step is what makes the from-scratch pipeline competitive with
COLMAP for thin-branch geometry. Section "Things I Care About" in the project
brief explicitly calls out the raw-vs-filtered side-by-side as the most
persuasive image in the whole project — `viz.plot_cloud_pair` produces that
figure, and this module provides the data behind it.

Mask convention (matches `segmentation.py`):
    mask[y, x] == True   → structure
    mask[y, x] == False  → sky
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FilterDiagnostics:
    """Per-point bookkeeping from a filtering pass — useful for tuning thresholds."""
    visible_count: np.ndarray     # (N,) int — frames in which the point projected in-bounds
    hit_count: np.ndarray         # (N,) int — frames where the projection landed on structure
    hit_rate: np.ndarray          # (N,) float in [0, 1] — hit_count / max(visible_count, 1)


# ---------------------------------------------------------------------------
# Projection helper. NOT from-scratch credit (it's just linear algebra) —
# the from-scratch credit is the filtering logic that uses it.
# ---------------------------------------------------------------------------

def project_points(P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points through a 3x4 projection matrix.

    Args:
        P: (3, 4) projection matrix.
        X: (N, 3) world-coordinate points.

    Returns:
        (uv, depth):
            uv:    (N, 2) pixel coords (x, y).
            depth: (N,)  z in the camera frame (positive = in front of cam).
    """
    X_h = np.hstack([X, np.ones((len(X), 1))])
    proj = X_h @ P.T              # (N, 3)
    depth = proj[:, 2]
    safe = np.where(np.abs(depth) > 1e-12, depth, 1e-12)
    uv = proj[:, :2] / safe[:, None]
    return uv, depth


# ---------------------------------------------------------------------------
# The from-scratch filter itself.
# ---------------------------------------------------------------------------

def filter_cloud_by_masks(
    points_3d: np.ndarray,
    projections: list[np.ndarray],
    masks: list[np.ndarray],
    hit_rate_threshold: float = 0.7,
    min_visible_frames: int = 4,
) -> tuple[np.ndarray, np.ndarray, FilterDiagnostics]:
    """Keep 3D points whose mask hit-rate across frames exceeds a threshold.

    For each 3D point X and each frame f:

        1. uv, depth = project(P_f, X)
        2. mark frame f as "visible" iff depth > 0 and uv is inside the image
        3. mark frame f as "hit" iff visible AND mask_f[round(uv)] == True

    Then hit_rate = hit_count / visible_count, and we keep points with

        hit_rate ≥ hit_rate_threshold   AND   visible_count ≥ min_visible_frames

    The min_visible_frames cutoff prevents lucky points seen in just one or
    two frames from being kept solely because they happened to project on
    structure those few times.

    Args:
        points_3d: (N, 3) world coordinates of the raw cloud.
        projections: list of F 3x4 projection matrices, one per frame.
        masks: list of F boolean masks (H, W), in the same order as
            `projections`. All masks must come from the same image resolution.
        hit_rate_threshold: minimum mask-hit fraction to keep a point.
        min_visible_frames: minimum number of frames in which the point must
            be visible for it to be eligible.

    Returns:
        (kept_points, kept_mask, diag) where:
            kept_points: (M, 3) the surviving subset of `points_3d`.
            kept_mask:   (N,) bool mask into the original cloud.
            diag:        FilterDiagnostics for inspection / tuning.
    """
    if len(projections) != len(masks):
        raise ValueError("projections and masks must be the same length.")
    if len(projections) == 0:
        raise ValueError("Need at least one frame to filter.")

    ### YOUR CODE HERE
    raise NotImplementedError("filter_cloud_by_masks: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# Optional auxiliary filters that fit naturally with the reprojection filter.
# Both are also FROM SCRATCH for credit.
# ---------------------------------------------------------------------------

def remove_statistical_outliers(
    points_3d: np.ndarray, k: int = 16, std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop points whose mean distance to k nearest neighbours is anomalously large.

    For each point, compute the mean distance to its k nearest neighbours;
    let μ and σ be the mean and std of those across the cloud. Keep points
    whose neighbourhood-mean is below μ + std_ratio·σ.

    Useful as a cheap "denoise" pass after the reprojection filter.

    Args:
        points_3d: (N, 3) cloud.
        k: number of nearest neighbours per point.
        std_ratio: cutoff in standard deviations above the mean.

    Returns:
        (kept_points, kept_mask) — kept_mask is (N,) bool.
    """
    ### YOUR CODE HERE
    raise NotImplementedError("remove_statistical_outliers: from-scratch implementation pending.")
    ### END YOUR CODE


def voxel_downsample(points_3d: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-grid downsample: replace each occupied voxel with the centroid.

    Args:
        points_3d: (N, 3) cloud.
        voxel_size: voxel edge length in the same units as the cloud.

    Returns:
        (M, 3) downsampled cloud where M ≤ N.
    """
    ### YOUR CODE HERE
    raise NotImplementedError("voxel_downsample: from-scratch implementation pending.")
    ### END YOUR CODE
