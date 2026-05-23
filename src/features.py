"""Stage 2 — SIFT feature detection and matching (library-wrapped).

Detect SIFT keypoints (Lowe, 2004), match across frame pairs with Lowe's ratio
test, and verify the matches geometrically with cv2's RANSAC F-matrix /
E-matrix estimation.

This module is *not* from-scratch: the from-scratch geometry lives in
`two_view.py`. The point of this module is to produce clean correspondences
so the from-scratch pipeline starts from a sane place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameFeatures:
    """SIFT keypoints + descriptors for a single image."""
    image_path: Path
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray   # (N, 128) float32

    @property
    def xy(self) -> np.ndarray:
        """Keypoint coordinates as an (N, 2) float array."""
        return np.array([kp.pt for kp in self.keypoints], dtype=np.float64)


@dataclass
class PairMatches:
    """Verified matches between two frames, with inlier mask."""
    pts1: np.ndarray         # (M, 2) — coords in frame 1
    pts2: np.ndarray         # (M, 2) — coords in frame 2
    inlier_mask: np.ndarray  # (M,) bool — True where RANSAC accepted the match
    F: np.ndarray | None     # 3x3 fundamental matrix (from cv2; reference)
    E: np.ndarray | None     # 3x3 essential matrix (from cv2; reference)

    @property
    def inlier_pts(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.inlier_mask
        return self.pts1[m], self.pts2[m]


def detect_sift(image_path: str | Path, max_features: int = 4000) -> FrameFeatures:
    """Run SIFT on a single image and return keypoints + descriptors.

    Args:
        image_path: path to an image file.
        max_features: cap on number of keypoints returned (sorted by response).

    Returns:
        FrameFeatures with keypoints (cv2.KeyPoint list) and descriptors (Nx128).
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    sift = cv2.SIFT_create(nfeatures=max_features)
    kps, desc = sift.detectAndCompute(img, None)
    if desc is None:
        desc = np.zeros((0, 128), dtype=np.float32)
    return FrameFeatures(image_path=image_path, keypoints=list(kps), descriptors=desc)


def lowe_ratio_match(
    feats1: FrameFeatures,
    feats2: FrameFeatures,
    ratio: float = 0.75,
    cross_check: bool = True,
) -> list[cv2.DMatch]:
    """Match SIFT descriptors with Lowe's ratio test (and optional cross-check).

    For each query descriptor we ask for its 2 nearest neighbours; we accept
    the match only if the best distance is < `ratio` × second-best distance
    (this is the standard "distinctive" test from Lowe 2004 §7.1).

    Args:
        feats1, feats2: features from two frames.
        ratio: Lowe's ratio threshold (0.7–0.8 is the usual sweet spot).
        cross_check: also require the match to be mutual (best both ways).

    Returns:
        List of cv2.DMatch entries that passed the test.
    """
    if len(feats1.descriptors) < 2 or len(feats2.descriptors) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(feats1.descriptors, feats2.descriptors, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]

    if cross_check:
        knn_rev = bf.knnMatch(feats2.descriptors, feats1.descriptors, k=2)
        rev_good = {n.trainIdx: n.queryIdx for n, _ in knn_rev}
        good = [m for m in good if rev_good.get(m.trainIdx) == m.queryIdx]

    return good


def matches_to_arrays(
    feats1: FrameFeatures,
    feats2: FrameFeatures,
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of DMatch into (pts1, pts2) numpy arrays of shape (M, 2)."""
    pts1 = np.array([feats1.keypoints[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts2 = np.array([feats2.keypoints[m.trainIdx].pt for m in matches], dtype=np.float64)
    return pts1, pts2


def verify_pair_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray | None = None,
    threshold_px: float = 1.0,
    confidence: float = 0.999,
) -> PairMatches:
    """Geometrically verify matches with cv2's RANSAC fundamental/essential estimation.

    If `K` is given we estimate the essential matrix (the geometry the
    from-scratch pipeline cares about); otherwise we estimate the fundamental
    matrix. Either way the output is a boolean inlier mask plus the matrix
    cv2 returned — useful as a reference to sanity-check the from-scratch
    implementation in `two_view.py`.

    Args:
        pts1, pts2: (M, 2) correspondences.
        K: 3x3 intrinsic matrix; if provided, estimates E.
        threshold_px: RANSAC reprojection threshold in pixels.
        confidence: RANSAC success probability.

    Returns:
        PairMatches bundle.
    """
    if len(pts1) < 8:
        raise ValueError(f"Need at least 8 matches for RANSAC; got {len(pts1)}.")

    F = E = None
    if K is not None:
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=confidence, threshold=threshold_px
        )
    else:
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, method=cv2.FM_RANSAC,
            ransacReprojThreshold=threshold_px, confidence=confidence,
        )

    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(pts1), bool)
    return PairMatches(pts1=pts1, pts2=pts2, inlier_mask=inliers, F=F, E=E)


def match_pair(
    image_path_a: str | Path,
    image_path_b: str | Path,
    K: np.ndarray | None = None,
    ratio: float = 0.75,
    threshold_px: float = 1.0,
) -> tuple[FrameFeatures, FrameFeatures, PairMatches]:
    """End-to-end: detect SIFT on both images, ratio-test, RANSAC-verify."""
    fa = detect_sift(image_path_a)
    fb = detect_sift(image_path_b)
    matches = lowe_ratio_match(fa, fb, ratio=ratio)
    if len(matches) < 8:
        raise RuntimeError(
            f"Only {len(matches)} ratio-test matches between {image_path_a} and "
            f"{image_path_b} — pair may be too far apart or too similar."
        )
    pts1, pts2 = matches_to_arrays(fa, fb, matches)
    pair = verify_pair_ransac(pts1, pts2, K=K, threshold_px=threshold_px)
    return fa, fb, pair
