"""Shared plotting utilities (library-wrapped).

All "paper-quality" figures are produced through these helpers so the visual
style stays consistent across notebooks. Output figures live in
`outputs/figures/`.

Convention used throughout the project:
    mask[y, x] == True   →  pixel is STRUCTURE (tree)
    mask[y, x] == False  →  pixel is SKY / background

This convention is documented in `segmentation.py` and enforced by both the
classical and SAM mask functions; see `filter_cloud.py` for how filtering
consumes it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  — registers the 3d projection

# Anchor at the project root (parent of src/) so figures land in the same
# outputs/figures/ regardless of whether the caller's cwd is the repo root
# (CLI runner) or notebooks/ (Jupyter). Previously this was a cwd-relative path,
# which silently wrote notebook figures to notebooks/outputs/figures/.
FIGDIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"


def _ensure_figdir() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: Figure, name: str, dpi: int = 150) -> Path:
    """Save a figure into outputs/figures/<name>.png and return its path."""
    _ensure_figdir()
    if not name.endswith((".png", ".pdf")):
        name = name + ".png"
    out = FIGDIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def show_image(img: np.ndarray, title: str = "", ax=None):
    """Display a BGR or grayscale image with the right colormap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    if img.ndim == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_matches(
    img1: np.ndarray, img2: np.ndarray,
    pts1: np.ndarray, pts2: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    title: str = "Matches",
    max_lines: int = 80,
) -> Figure:
    """Side-by-side image pair with match lines (green = inlier, red = outlier)."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1 + w2] = img2 if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

    n = len(pts1)
    step = max(1, n // max_lines)
    if inlier_mask is None:
        inlier_mask = np.ones(n, dtype=bool)

    for i in range(0, n, step):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        color = "lime" if inlier_mask[i] else "red"
        ax.plot([x1, x2 + w1], [y1, y2], color=color, linewidth=0.6, alpha=0.7)
        ax.scatter([x1, x2 + w1], [y1, y2], color=color, s=4)
    return fig


def plot_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    title: str = "Point cloud",
    elev: float = 20, azim: float = -60,
    s: float = 1.5,
) -> Figure:
    """Render a 3D scatter of an (N,3) point cloud."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = rgb.astype(float) / 255.0 if rgb is not None else "tab:blue"
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=s, depthshade=False)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    _equalize_3d_axes(ax, xyz)
    return fig


def plot_cloud_pair(
    xyz_raw: np.ndarray,
    xyz_filtered: np.ndarray,
    title_raw: str = "Raw cloud",
    title_filtered: str = "Mask-filtered cloud",
    suptitle: str = "Reprojection-based filtering",
) -> Figure:
    """Side-by-side 3D scatter for the raw-vs-filtered cloud comparison.

    This is the single most persuasive figure in the project — the visible drop
    in noise and emergence of branch structure after filtering. Lives here so
    every ablation reproduces it the same way.
    """
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.scatter(xyz_raw[:, 0], xyz_raw[:, 1], xyz_raw[:, 2],
                s=1.5, c="tab:gray", depthshade=False)
    ax2.scatter(xyz_filtered[:, 0], xyz_filtered[:, 1], xyz_filtered[:, 2],
                s=1.5, c="tab:green", depthshade=False)
    ax1.set_title(f"{title_raw} (N={len(xyz_raw)})")
    ax2.set_title(f"{title_filtered} (N={len(xyz_filtered)})")
    for ax, pts in [(ax1, xyz_raw), (ax2, xyz_filtered)]:
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        _equalize_3d_axes(ax, pts)
    fig.suptitle(suptitle)
    return fig


def plot_mask_overlay(
    image: np.ndarray, mask: np.ndarray, title: str = "",
    alpha: float = 0.45, color=(0, 255, 0), ax=None,
):
    """Overlay a boolean structure mask on top of a BGR image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else \
        cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color)
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ a is parallel to b (Rodrigues' formula).

    Used to bring the recovered trunk axis onto plot-vertical. Pure SfM has no
    gravity reference, so COLMAP's world frame is arbitrary (here ~aligned with
    the tilted capture camera) and a vertical trunk is stored tilted. This is a
    rigid rotation: it changes the *view*, never relative angles or distances.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-8:                       # already (anti)parallel
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def plot_branch_graph(
    nodes: np.ndarray, edges: list[tuple[int, int]],
    trunk_axis: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Branch graph",
    align_trunk_vertical: bool = True,
) -> Figure:
    """3D scatter of skeleton nodes with edges drawn as line segments.

    If `align_trunk_vertical` and a trunk axis is given, the cloud is rigidly
    rotated about the trunk root so the trunk points up the +Z (plot-vertical)
    axis — undoing COLMAP's arbitrary world orientation for display only.
    """
    if trunk_axis is not None and align_trunk_vertical:
        p0, d = trunk_axis
        R = rotation_aligning(d, np.array([0.0, 0.0, 1.0]))
        nodes = (R @ (nodes - p0).T).T + p0
        trunk_axis = (p0, R @ np.asarray(d, float))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="tab:brown", s=3)
    for i, j in edges:
        x = [nodes[i, 0], nodes[j, 0]]
        y = [nodes[i, 1], nodes[j, 1]]
        z = [nodes[i, 2], nodes[j, 2]]
        ax.plot(x, y, z, c="saddlebrown", linewidth=0.6)
    if trunk_axis is not None:
        p0, d = trunk_axis
        ts = np.linspace(-2, 2, 50)
        trunk = p0[None, :] + ts[:, None] * d[None, :]
        ax.plot(trunk[:, 0], trunk[:, 1], trunk[:, 2],
                c="tab:red", linewidth=2.0, label="trunk axis")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    _equalize_3d_axes(ax, nodes)
    return fig


def _equalize_3d_axes(ax, pts: np.ndarray) -> None:
    """Make a 3D axes use equal aspect so the cloud isn't distorted."""
    if len(pts) == 0:
        return
    mins = pts.min(axis=0); maxs = pts.max(axis=0)
    span = (maxs - mins).max() / 2
    center = (maxs + mins) / 2
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
