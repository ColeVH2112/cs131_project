"""Stage 1 — camera calibration and frame extraction (library-wrapped).

Two responsibilities:

1. Recover the camera intrinsic matrix K from a checkerboard video using cv2's
   standard pinhole calibration (Zhang's method, Lec. 6).
2. Extract frames from the skyward orbital capture at a fixed stride and write
   them to disk for downstream stages.

Nothing here is required to be from-scratch — these are I/O / OpenCV
glue utilities so the from-scratch pipeline (two_view, filtering, trunk,
branch graph) has clean inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# Default checkerboard layout. Override at call time if your printed pattern differs.
DEFAULT_PATTERN_SIZE = (9, 6)   # interior corners (cols, rows)
DEFAULT_SQUARE_SIZE_M = 0.025   # 25 mm squares


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics bundled with the image size they were calibrated at."""
    K: np.ndarray          # 3x3 camera matrix
    dist: np.ndarray       # (5,) distortion coefficients [k1, k2, p1, p2, k3]
    image_size: tuple[int, int]   # (W, H)
    rms: float             # reprojection RMS from calibration

    def save(self, path: str | Path) -> None:
        np.savez(path, K=self.K, dist=self.dist,
                 image_size=np.asarray(self.image_size), rms=self.rms)

    @classmethod
    def load(cls, path: str | Path) -> "CameraIntrinsics":
        data = np.load(path)
        return cls(
            K=data["K"],
            dist=data["dist"],
            image_size=tuple(int(x) for x in data["image_size"]),
            rms=float(data["rms"]),
        )


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    stride: int = 5,
    max_frames: int | None = None,
    prefix: str = "frame",
) -> list[Path]:
    """Decode a video and write every `stride`-th frame as a PNG.

    Args:
        video_path: input .mov / .mp4 file.
        out_dir: directory to write frames into (created if missing).
        stride: keep one frame out of every `stride` decoded frames.
        max_frames: optional cap on number of frames written.
        prefix: filename prefix; frames are written as `<prefix>_<idx:05d>.png`.

    Returns:
        List of written frame paths, in capture order.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    written: list[Path] = []
    idx = 0
    kept = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                out_path = out_dir / f"{prefix}_{kept:05d}.png"
                cv2.imwrite(str(out_path), frame)
                written.append(out_path)
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    return written


def _detect_checkerboard_corners(
    image: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """Find sub-pixel checkerboard corners in a single image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def calibrate_from_images(
    image_paths: Iterable[str | Path],
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    square_size: float = DEFAULT_SQUARE_SIZE_M,
) -> CameraIntrinsics:
    """Recover K and distortion from a folder of checkerboard images.

    Uses cv2.calibrateCamera, which implements Zhang's planar calibration method.

    Args:
        image_paths: iterable of paths to checkerboard images (e.g. extracted frames).
        pattern_size: (cols, rows) interior corners of the checkerboard.
        square_size: physical edge length of one square, in metres.

    Returns:
        CameraIntrinsics with K, distortion, image size, and RMS reprojection error.
    """
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        raise ValueError("No checkerboard images provided.")

    # 3D object points template (planar checkerboard at Z=0).
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    n_found = 0

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if image_size is None:
            h, w = img.shape[:2]
            image_size = (w, h)
        found, corners = _detect_checkerboard_corners(img, pattern_size)
        if found:
            object_points.append(objp)
            image_points.append(corners)
            n_found += 1

    if n_found < 8:
        raise RuntimeError(
            f"Only {n_found} checkerboard detections — need at least 8. "
            "Capture a longer / more varied calibration clip."
        )

    rms, K, dist, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    return CameraIntrinsics(
        K=np.asarray(K),
        dist=np.asarray(dist).ravel(),
        image_size=image_size,
        rms=float(rms),
    )


def calibrate_from_video(
    video_path: str | Path,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    square_size: float = DEFAULT_SQUARE_SIZE_M,
    stride: int = 10,
    max_frames: int = 40,
    tmp_dir: str | Path = "data/calibration/_frames",
) -> CameraIntrinsics:
    """Convenience: extract frames from a calibration video and calibrate."""
    frames = extract_frames(video_path, tmp_dir, stride=stride, max_frames=max_frames,
                            prefix="calib")
    return calibrate_from_images(frames, pattern_size, square_size)


def undistort_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Apply lens undistortion to a single image."""
    return cv2.undistort(image, intrinsics.K, intrinsics.dist)
