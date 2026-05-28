"""Stage 3b — COLMAP multi-view Structure-from-Motion wrapper (library-wrapped).

Runs COLMAP's standard automatic_reconstructor pipeline:

    1. Feature extraction (SIFT)
    2. Exhaustive feature matching
    3. Incremental mapping (mapper)

This is intentionally library-wrapped — the from-scratch geometric work
lives in `two_view.py`. COLMAP here is the "full multi-view" reference that
the two-view-vs-multi-view ablation compares against.

Requires the `colmap` CLI on PATH. See README for install instructions.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SparseReconstruction:
    """A COLMAP sparse model: 3D points + per-image camera poses.

    Pose convention follows COLMAP: world-to-camera, quaternion + translation.
    """
    points_3d: np.ndarray              # (N, 3) world coordinates
    point_colors: np.ndarray | None    # (N, 3) uint8 RGB or None
    image_names: list[str]             # one entry per registered image
    quaternions: np.ndarray            # (M, 4) w,x,y,z  — world-to-camera rotation
    translations: np.ndarray           # (M, 3)         — world-to-camera translation
    K: np.ndarray | None               # 3x3 if all images share intrinsics, else None


def _ensure_colmap_available() -> None:
    if shutil.which("colmap") is None:
        raise RuntimeError(
            "`colmap` was not found on PATH. Install COLMAP (see README) and "
            "make sure the binary is reachable, then re-run."
        )


def run_colmap(
    image_dir: str | Path,
    workspace_dir: str | Path,
    quality: str = "medium",
    single_camera: bool = True,
    use_gpu: bool = True,
    mask_dir: str | Path | None = None,
    matcher: str = "sequential",
) -> Path:
    """Run COLMAP's three-step SfM pipeline on a directory of frames.

    Internally calls `feature_extractor` → matcher → `mapper`, rather than
    `automatic_reconstructor`, so we can pass `--ImageReader.mask_path` to
    restrict feature extraction to a subset of each frame (e.g. the
    bark-mask region for tree captures). This is the supported COLMAP way
    to keep the extractor away from sky/canopy/background pixels.

    Args:
        image_dir: directory containing extracted frames for one tree.
        workspace_dir: directory COLMAP will write the database and `sparse/`
            model into. Created if missing.
        quality: rough preset mapped to `SiftExtraction.max_num_features`
            (low → 4096, medium → 8192, high → 16384, extreme → 32768).
        single_camera: assume all frames share intrinsics (true for one phone clip).
        use_gpu: pass `--SiftExtraction.use_gpu 1` and `--SiftMatching.use_gpu 1`.
            On macOS Homebrew COLMAP this uses the OpenGL SiftGPU path, which
            is more stable than the CPU matcher in our experience.
        mask_dir: optional directory of binary PNG masks. For each frame
            `image_dir/<name>.<ext>`, COLMAP looks for `mask_dir/<name>.png`
            (extension always `.png`, basename matches the image). White
            pixels (255) → use for SIFT extraction, black (0) → ignore.
        matcher: "sequential" matches each frame against a small window of
            its neighbours (right choice for a video orbit — much faster
            than exhaustive, and avoids the macOS Homebrew exhaustive_matcher
            crash mode). "exhaustive" matches every pair.

    Returns:
        Path to the produced sparse model directory (`<workspace>/sparse/0/`).
    """
    _ensure_colmap_available()

    image_dir = Path(image_dir)
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if not any(image_dir.iterdir()):
        raise RuntimeError(f"No images found in {image_dir}.")

    database = workspace_dir / "database.db"
    sparse_root = workspace_dir / "sparse"
    sparse_root.mkdir(exist_ok=True)

    gpu_flag = "1" if use_gpu else "0"
    max_features = {
        "low":     4096,
        "medium":  8192,
        "high":    16384,
        "extreme": 32768,
    }.get(quality, 8192)

    # 1. Feature extraction.
    extract_cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path",    str(image_dir),
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--ImageReader.camera_model",  "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu",          gpu_flag,
        "--SiftExtraction.max_num_features", str(max_features),
    ]
    if mask_dir is not None:
        mask_dir = Path(mask_dir)
        if not mask_dir.exists():
            raise FileNotFoundError(f"mask_dir does not exist: {mask_dir}")
        extract_cmd.extend(["--ImageReader.mask_path", str(mask_dir)])
    subprocess.run(extract_cmd, check=True)

    # 2. Feature matching.
    if matcher == "sequential":
        match_cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(database),
            "--SiftMatching.use_gpu", gpu_flag,
        ]
    elif matcher == "exhaustive":
        match_cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database),
            "--SiftMatching.use_gpu", gpu_flag,
        ]
    else:
        raise ValueError(f"matcher must be 'sequential' or 'exhaustive', got {matcher!r}")
    subprocess.run(match_cmd, check=True)

    # 3. Incremental mapping (sparse reconstruction).
    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", str(database),
        "--image_path",    str(image_dir),
        "--output_path",   str(sparse_root),
    ]
    subprocess.run(mapper_cmd, check=True)

    submodels = sorted([p for p in sparse_root.iterdir() if p.is_dir()])
    if not submodels:
        raise RuntimeError(
            "COLMAP mapper produced no sub-models — check that feature "
            "extraction and matching succeeded, and that the masks (if any) "
            "didn't reject every frame."
        )
    return submodels[0]


# ---------------------------------------------------------------------------
# COLMAP binary model readers.
# Format spec: https://colmap.github.io/format.html
# ---------------------------------------------------------------------------

def _read_next_bytes(fid, num_bytes, format_char_sequence, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + format_char_sequence, data)


def _read_points3D_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP's points3D.bin → (xyz Nx3, rgb Nx3 uint8)."""
    xyzs, rgbs = [], []
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            _, x, y, z, r, g, b, _ = _read_next_bytes(f, 43, "QdddBBBd")
            track_len = _read_next_bytes(f, 8, "Q")[0]
            f.read(8 * track_len)   # (image_id, point2D_idx) pairs, ignored
            xyzs.append([x, y, z])
            rgbs.append([r, g, b])
    return np.asarray(xyzs, dtype=np.float64), np.asarray(rgbs, dtype=np.uint8)


def _read_images_bin(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read COLMAP's images.bin → (names, quats Nx4 w,x,y,z, t Nx3)."""
    names, quats, ts = [], [], []
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            _, qw, qx, qy, qz, tx, ty, tz, _ = _read_next_bytes(f, 64, "idddddddi")
            # Image name is a null-terminated string.
            chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                chars.append(c)
            names.append(b"".join(chars).decode("utf-8"))
            n_pts2d = _read_next_bytes(f, 8, "Q")[0]
            f.read(24 * n_pts2d)    # (x, y, point3D_id) per 2D observation
            quats.append([qw, qx, qy, qz])
            ts.append([tx, ty, tz])
    return names, np.asarray(quats), np.asarray(ts)


def _read_cameras_bin(path: Path) -> dict[int, np.ndarray]:
    """Read COLMAP's cameras.bin and return {camera_id: 3x3 K}.

    Only handles PINHOLE / SIMPLE_PINHOLE / SIMPLE_RADIAL — fine for phone clips.
    """
    cameras = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            cam_id, model_id, w, h = _read_next_bytes(f, 24, "iiQQ")
            # Number of params depends on model. SIMPLE_PINHOLE=3, PINHOLE=4,
            # SIMPLE_RADIAL=4, RADIAL=5. We read until we have enough for K.
            if model_id == 0:           # SIMPLE_PINHOLE: f, cx, cy
                f_, cx, cy = _read_next_bytes(f, 24, "ddd")
                K = np.array([[f_, 0, cx], [0, f_, cy], [0, 0, 1.0]])
            elif model_id == 1:         # PINHOLE: fx, fy, cx, cy
                fx, fy, cx, cy = _read_next_bytes(f, 32, "dddd")
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            elif model_id == 2:         # SIMPLE_RADIAL: f, cx, cy, k
                f_, cx, cy, _k = _read_next_bytes(f, 32, "dddd")
                K = np.array([[f_, 0, cx], [0, f_, cy], [0, 0, 1.0]])
            else:
                # Fall back: skip the rest of this camera's params by reading
                # COLMAP's documented param counts. Caller will get K=None.
                raise NotImplementedError(
                    f"Camera model id {model_id} not supported by this reader."
                )
            cameras[cam_id] = K
    return cameras


def load_sparse_model(model_dir: str | Path) -> SparseReconstruction:
    """Load a COLMAP sparse model from binary files into a SparseReconstruction."""
    model_dir = Path(model_dir)
    points_bin = model_dir / "points3D.bin"
    images_bin = model_dir / "images.bin"
    cameras_bin = model_dir / "cameras.bin"

    if not (points_bin.exists() and images_bin.exists() and cameras_bin.exists()):
        raise FileNotFoundError(
            f"Expected points3D.bin / images.bin / cameras.bin in {model_dir}."
        )

    xyz, rgb = _read_points3D_bin(points_bin)
    names, quats, ts = _read_images_bin(images_bin)
    cams = _read_cameras_bin(cameras_bin)
    # If all images share one camera, return that K; otherwise None.
    K = next(iter(cams.values())) if len(cams) == 1 else None

    return SparseReconstruction(
        points_3d=xyz,
        point_colors=rgb,
        image_names=names,
        quaternions=quats,
        translations=ts,
        K=K,
    )


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def projection_matrices(recon: SparseReconstruction) -> list[np.ndarray]:
    """Per-image 3x4 projection matrix P = K @ [R | t] (world → image)."""
    if recon.K is None:
        raise ValueError("Multiple cameras in the model; pass intrinsics explicitly.")
    Ps = []
    for q, t in zip(recon.quaternions, recon.translations):
        R = quat_to_R(q)
        Ps.append(recon.K @ np.hstack([R, t.reshape(3, 1)]))
    return Ps


def save_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    """Write a point cloud to a minimal binary-free PLY file (ASCII)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    has_color = rgb is not None
    header = [
        "ply", "format ascii 1.0", f"element vertex {n}",
        "property float x", "property float y", "property float z",
    ]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(n):
            x, y, z = xyz[i]
            if has_color:
                r, g, b = rgb[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x} {y} {z}\n")


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an ASCII PLY written by `save_ply` back into (xyz, rgb-or-None)."""
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()
    header_end = next(i for i, ln in enumerate(lines) if ln.strip() == "end_header")
    n = int(next(ln for ln in lines[:header_end] if ln.startswith("element vertex")).split()[-1])
    has_color = any("red" in ln for ln in lines[:header_end])
    xyz = np.zeros((n, 3))
    rgb = np.zeros((n, 3), dtype=np.uint8) if has_color else None
    for i, ln in enumerate(lines[header_end + 1 : header_end + 1 + n]):
        parts = ln.split()
        xyz[i] = [float(p) for p in parts[:3]]
        if has_color:
            rgb[i] = [int(p) for p in parts[3:6]]
    return xyz, rgb
