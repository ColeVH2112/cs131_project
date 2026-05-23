"""Stage 3a — from-scratch two-view geometry (FROM SCRATCH).

This module is the heart of the CS 131 geometric content. Every function below
has a complete signature and docstring but an empty body marked with
`### YOUR CODE HERE` / `### END YOUR CODE`. Graders look for those markers to
distinguish my code from library code, so do not delete them.

The pipeline implemented across these functions is:

    normalize_points
        → centroid to origin, mean distance to sqrt(2)
        (Hartley's normalisation — see HZ §11.2)

    eight_point_algorithm
        → solve  x2ᵀ F x1 = 0  for F via SVD of the linear system,
          then enforce det(F) = 0 by zeroing the smallest singular value

    fundamental_to_essential
        → E = K2ᵀ F K1, then re-project to the valid essential manifold
          (singular values {σ, σ, 0})

    decompose_essential
        → SVD of E → four candidate (R, t) hypotheses
          (HZ §9.6.2: W ≡ [[0,-1,0],[1,0,0],[0,0,1]])

    triangulate_dlt
        → linear DLT: stack two cross-product rows per camera, solve via SVD

    cheirality_check
        → for each (R, t) candidate, triangulate and count how many points
          lie in front of both cameras (positive depth in both); pick the winner

    two_view_reconstruction
        → glue function: takes correspondences + K, returns (R, t, X) ready to
          compare against cv2.findEssentialMat / cv2.recoverPose
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TwoViewResult:
    """Output of a complete two-view reconstruction."""
    R: np.ndarray            # 3x3 rotation, world(=cam1) → cam2
    t: np.ndarray            # 3-vector translation, unit length (scale-ambiguous)
    points_3d: np.ndarray    # (M, 3) triangulated points in cam1 coords
    inlier_mask: np.ndarray  # (N,) bool — True for cheirality-positive points
    E: np.ndarray            # 3x3 essential matrix used
    F: np.ndarray | None     # 3x3 fundamental matrix (if computed)


# ---------------------------------------------------------------------------
# Hartley normalization
# ---------------------------------------------------------------------------

def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: translate centroid to origin, scale so the
    mean distance from the origin is sqrt(2).

    Returns the normalised homogeneous points and the 3x3 similarity matrix T
    such that  normalised = (T @ homogeneous.T).T. This T is needed later to
    "denormalise" the resulting F via  F = T2ᵀ F_norm T1.

    Args:
        points: (N, 2) array of pixel coordinates.

    Returns:
        (points_norm_h, T) — points_norm_h is (N, 3) in homogeneous coords;
        T is the 3x3 normalisation matrix that produced them.
    """
    ### YOUR CODE HERE
    raise NotImplementedError("normalize_points: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# Normalised 8-point algorithm  →  F
# ---------------------------------------------------------------------------

def eight_point_algorithm(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Solve for the fundamental matrix F from N>=8 correspondences.

    Each correspondence gives one linear equation in the 9 unknowns of F via
    x2ᵀ F x1 = 0. Stack into Ax = 0 (A is N×9), take the right null vector
    (smallest singular value of A) as f, reshape into 3x3, and enforce
    det(F) = 0 by zeroing the smallest singular value of the resulting matrix.

    Inputs should be normalised first via `normalize_points` for numerical
    stability — this function applies that normalisation internally and
    denormalises before returning F.

    Args:
        pts1: (N, 2) pixel coords in image 1.
        pts2: (N, 2) pixel coords in image 2.

    Returns:
        3x3 fundamental matrix F (defined up to scale, det(F) = 0).
    """
    if len(pts1) < 8:
        raise ValueError(f"Need at least 8 correspondences; got {len(pts1)}.")
    if len(pts1) != len(pts2):
        raise ValueError("pts1 and pts2 must have the same length.")

    ### YOUR CODE HERE
    raise NotImplementedError("eight_point_algorithm: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# F  →  E  →  (R, t) candidates
# ---------------------------------------------------------------------------

def fundamental_to_essential(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Convert F → E using the calibration matrices.

    E = K2ᵀ F K1. Then re-project the result onto the essential-matrix
    manifold by enforcing singular values (σ, σ, 0). Any non-zero σ is fine;
    we typically pick σ = (σ1 + σ2) / 2.

    Args:
        F: 3x3 fundamental matrix.
        K1, K2: 3x3 intrinsic matrices for cameras 1 and 2.

    Returns:
        3x3 essential matrix E with rank 2 and equal non-zero singular values.
    """
    ### YOUR CODE HERE
    raise NotImplementedError("fundamental_to_essential: from-scratch implementation pending.")
    ### END YOUR CODE


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Recover the four (R, t) candidates from an essential matrix.

    Per HZ §9.6.2, given E = U diag(1,1,0) Vᵀ and
        W = [[0,-1,0],[1, 0,0],[0, 0,1]]
    the candidate rotations are R = U W Vᵀ and R = U Wᵀ Vᵀ (correcting
    each for det = +1 by negating if necessary), and the candidate
    translations are ±u3 (the third column of U).

    Args:
        E: 3x3 essential matrix.

    Returns:
        List of 4 (R, t) tuples. Translation is unit-length (scale is
        unrecoverable from two views without external metric information).
    """
    ### YOUR CODE HERE
    raise NotImplementedError("decompose_essential: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# DLT triangulation
# ---------------------------------------------------------------------------

def triangulate_dlt(
    P1: np.ndarray, P2: np.ndarray,
    pts1: np.ndarray, pts2: np.ndarray,
) -> np.ndarray:
    """Linear triangulation by Direct Linear Transform.

    For each correspondence (x1, x2), the projection equations give
        x1 × (P1 X) = 0    and    x2 × (P2 X) = 0
    Each cross-product yields two linearly-independent rows (the third is a
    linear combination of the other two). Stacking four rows per point gives
    A X = 0 (A is 4×4), and X is the right null vector (smallest singular
    value of A). Dehomogenise by dividing by X[3].

    Args:
        P1, P2: 3x4 projection matrices for cameras 1 and 2 (P = K [R | t]).
        pts1, pts2: (N, 2) corresponding pixel coords in each image.

    Returns:
        (N, 3) array of triangulated 3D points in the world frame of P1.
    """
    if len(pts1) != len(pts2):
        raise ValueError("pts1 and pts2 must have the same length.")

    ### YOUR CODE HERE
    raise NotImplementedError("triangulate_dlt: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# Cheirality check  →  pick the geometrically valid (R, t) from the 4 candidates
# ---------------------------------------------------------------------------

def cheirality_check(
    candidates: list[tuple[np.ndarray, np.ndarray]],
    pts1: np.ndarray, pts2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the (R, t) hypothesis that puts the most points in front of
    both cameras.

    For each candidate (R, t):
        1. Build P1 = K [I | 0] and P2 = K [R | t].
        2. Triangulate all correspondences via `triangulate_dlt`.
        3. Count the points X with positive depth in *both* cameras:
            cam1 depth = X_z  (with cam1 at origin)
            cam2 depth = (R X + t)_z
        4. Keep the candidate with the highest count.

    "Cheirality" comes from "handedness" — only one of the four sign-flipped
    decompositions corresponds to a real-world geometry where the scene is
    actually in front of both cameras.

    Args:
        candidates: list of 4 (R, t) tuples from `decompose_essential`.
        pts1, pts2: (N, 2) correspondences.
        K: 3x3 intrinsic matrix (assumed shared between the two views).

    Returns:
        (R, t, points_3d, mask) where:
            R, t:        the winning candidate
            points_3d:   (N, 3) triangulated under the winner
            mask:        (N,) bool of cheirality-positive points
    """
    ### YOUR CODE HERE
    raise NotImplementedError("cheirality_check: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# End-to-end glue
# ---------------------------------------------------------------------------

def two_view_reconstruction(
    pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray,
) -> TwoViewResult:
    """Full from-scratch two-view reconstruction.

    Pipeline:
        F = eight_point_algorithm(pts1, pts2)
        E = fundamental_to_essential(F, K, K)
        candidates = decompose_essential(E)
        R, t, X, mask = cheirality_check(candidates, pts1, pts2, K)

    This is the function the milestone notebook compares against
    cv2.findEssentialMat + cv2.recoverPose. The expected differences are tiny
    rotational disagreements (degree-fractions) and a scale that matches up to
    sign — translation is only recoverable up to a positive scalar from two
    views.

    Args:
        pts1, pts2: (N, 2) verified correspondences (run RANSAC first).
        K: 3x3 intrinsic matrix.

    Returns:
        TwoViewResult bundle.
    """
    ### YOUR CODE HERE
    raise NotImplementedError("two_view_reconstruction: from-scratch implementation pending.")
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# Diagnostics — helpers for the milestone notebook to sanity-check the result.
# These are NOT from-scratch credit; they just present errors nicely.
# ---------------------------------------------------------------------------

def sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """First-order approximation to reprojection error given F.

    Useful for evaluating an F estimate without doing a full nonlinear fit:
        d_S(x1, x2; F) = (x2ᵀ F x1)² / (|| (F x1)_{1:2} ||² + || (Fᵀ x2)_{1:2} ||²)

    Args:
        F: 3x3 fundamental matrix.
        pts1, pts2: (N, 2) correspondences.

    Returns:
        (N,) Sampson distance per correspondence (in pixels²).
    """
    x1 = np.hstack([pts1, np.ones((len(pts1), 1))])
    x2 = np.hstack([pts2, np.ones((len(pts2), 1))])
    Fx1 = x1 @ F.T            # (N, 3)
    Ftx2 = x2 @ F             # (N, 3)
    num = np.einsum("ij,jk,ik->i", x2, F, x1) ** 2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / (denom + 1e-12)


def reprojection_error(P: np.ndarray, X: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Per-point reprojection error in pixels.

    Args:
        P: 3x4 projection matrix.
        X: (N, 3) 3D points.
        pts: (N, 2) observed image coords.

    Returns:
        (N,) Euclidean pixel error per point.
    """
    X_h = np.hstack([X, np.ones((len(X), 1))])
    proj = (P @ X_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return np.linalg.norm(proj - pts, axis=1)
