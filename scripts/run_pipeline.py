#!/usr/bin/env python
"""End-to-end pipeline: skyward orbital video → annotated tree branch graph.

This is the deployment script. It runs the same stages the notebooks do, but
in a single Python process with no Jupyter overhead. Use this when you want
the final artefact and don't need the per-stage notebook intermediate displays.

Usage:
    python scripts/run_pipeline.py <tree_id> [options]

Examples:
    # Calibrated run with real-world trunk height (gives metric output):
    python scripts/run_pipeline.py tree_5 --measured-trunk-height 3.5

    # Quick run at medium quality:
    python scripts/run_pipeline.py tree_5 --quality medium

    # Re-run skipping frame extraction (frames already extracted):
    python scripts/run_pipeline.py tree_5  # reuses cached frames/CLAHE automatically

Prerequisites:
    1. data/calibration/intrinsics.npz must exist (run notebook 01 once)
    2. data/captures/<tree_id>.mov must exist (the skyward orbit video)

Outputs:
    outputs/reconstructions/<tree_id>_sparse.ply        # raw COLMAP cloud
    outputs/reconstructions/<tree_id>_filtered.ply      # after reprojection filter
    outputs/reconstructions/<tree_id>_graph.npz         # branch graph (nodes + edges + parents)
    outputs/metrics/<tree_id>_predictions.csv           # per-branch attachment height + angle
    outputs/figures/pipeline_<tree_id>.png              # final branch-graph figure
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

# Make src/ importable regardless of cwd.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from src import (  # noqa: E402
    branch_graph,
    calibration,
    evaluate,
    filter_cloud,
    segmentation,
    sfm,
    trunk,
    viz,
)


def banner(title: str) -> None:
    print()
    print(f"=== {title} ===")


def stage_extract_frames(capture: Path, raw_frames: Path, stride: int, max_frames: int) -> int:
    """Stage 1: extract frames from the source video (skipped if already cached)."""
    raw_frames.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw_frames.glob("frame_*.png"))
    if existing:
        print(f"Reusing {len(existing)} extracted frames in {raw_frames}")
        return len(existing)
    if not capture.exists():
        raise FileNotFoundError(
            f"Capture video not found at {capture}. "
            "Record a skyward orbit and save it there."
        )
    frames = calibration.extract_frames(
        capture, raw_frames, stride=stride, max_frames=max_frames,
    )
    print(f"Extracted {len(frames)} frames → {raw_frames}")
    return len(frames)


def stage_clahe(raw_frames: Path, clahe_frames: Path) -> None:
    """Stage 2: CLAHE on the LAB L channel (skipped if already cached)."""
    clahe_frames.mkdir(parents=True, exist_ok=True)
    src = sorted(raw_frames.glob("frame_*.png"))
    if len(list(clahe_frames.glob("frame_*.png"))) >= len(src):
        print(f"Reusing existing CLAHE frames in {clahe_frames}")
        return
    op = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    for p in src:
        img = cv2.imread(str(p))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = op.apply(lab[:, :, 0])
        cv2.imwrite(str(clahe_frames / p.name), cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    print(f"CLAHE-enhanced {len(src)} frames → {clahe_frames}")


def stage_colmap(clahe_frames: Path, workspace: Path, quality: str) -> sfm.SparseReconstruction:
    """Stage 3: COLMAP multi-view sparse reconstruction."""
    shutil.rmtree(workspace, ignore_errors=True)
    sparse_dir = sfm.run_colmap(
        image_dir=clahe_frames,
        workspace_dir=workspace,
        matcher="sequential",
        quality=quality,
        single_camera=True,
    )
    recon = sfm.load_sparse_model(sparse_dir)
    print(f"Registered {len(recon.image_names)} cameras, {len(recon.points_3d)} points")
    return recon


def stage_mask_and_filter(
    recon: sfm.SparseReconstruction,
    clahe_frames: Path,
    hit_rate_threshold: float,
    min_visible_frames: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Stage 4 + 5: per-frame bark masks + reprojection filter + outlier removal."""
    # Build name → path index for matching reconstruction images to disk frames.
    name_to_path = {p.name: p for p in sorted(clahe_frames.glob("frame_*.png"))}

    masks: list[np.ndarray] = []
    for name in recon.image_names:
        img = cv2.imread(str(name_to_path[name]))
        mask_float = segmentation.bark_color_mask_lab(
            img,
            bark_a_min=128, bark_b_min=130,
            texture_weight=0.3,
            spatial_sigma_frac=0.25,
        )
        masks.append(mask_float > 0.5)

    projections = sfm.projection_matrices(recon)
    kept_pts, _, _ = filter_cloud.filter_cloud_by_masks(
        recon.points_3d, projections, masks,
        hit_rate_threshold=hit_rate_threshold,
        min_visible_frames=min_visible_frames,
    )
    n_pre = len(recon.points_3d)
    print(f"Reprojection filter kept {len(kept_pts)} / {n_pre} ({100*len(kept_pts)/n_pre:.1f}%)")

    kept_clean, _ = filter_cloud.remove_statistical_outliers(kept_pts, k=16, std_ratio=2.0)
    print(f"After statistical outlier removal: {len(kept_clean)}")
    return kept_clean, masks


def stage_skeleton(
    points: np.ndarray,
    vertical_axis: int,
    k: int,
    max_edge_length: float,
    min_spur_length: float,
    collinear_tolerance_deg: float,
) -> tuple[trunk.TrunkFit, branch_graph.BranchGraph]:
    """Stage 6 + 7: RANSAC trunk + Dijkstra branch graph."""
    fit = trunk.ransac_trunk_axis(
        points, num_iter=2000, inlier_thresh=0.05,
        vertical_prior=True, vertical_axis=vertical_axis,
        rng=np.random.default_rng(131),
    )
    print(f"Trunk: root={fit.root}, direction={fit.direction}")
    print(f"Trunk extent (COLMAP units): {trunk.trunk_height(fit, points):.2f}")

    graph = branch_graph.extract_branch_graph(
        points, fit.root,
        k=k, max_edge_length=max_edge_length,
        min_spur_length=min_spur_length,
        collinear_tolerance_deg=collinear_tolerance_deg,
    )
    print(f"Branch graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return fit, graph


def stage_measurements(
    fit: trunk.TrunkFit,
    graph: branch_graph.BranchGraph,
    cloud: np.ndarray,
    measured_trunk_height_m: float | None,
) -> tuple[list[evaluate.PredBranch], float | None, str]:
    """Stage 8: extract per-branch measurements, with optional metric scaling."""
    preds = evaluate.predicted_branches_from_graph(
        nodes=graph.nodes, edges=graph.edges,
        trunk_root=fit.root, trunk_axis=fit.direction,
        parent_of=graph.parent_of,
    )

    if measured_trunk_height_m is None:
        return preds, None, "units"

    trunk_extent = trunk.trunk_height(fit, cloud)
    if trunk_extent <= 0:
        print("WARNING: trunk extent is zero — cannot scale. Reporting in COLMAP units.")
        return preds, None, "units"

    scale = measured_trunk_height_m / trunk_extent
    print(f"Scale calibration: 1 COLMAP unit = {scale:.4f} m")
    scaled = [
        evaluate.PredBranch(
            branch_id=p.branch_id,
            attach_height_m=p.attach_height_m * scale,
            attach_angle_deg=p.attach_angle_deg,
        )
        for p in preds
    ]
    return scaled, scale, "m"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full skyward-orbital tree-reconstruction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tree_id", help="Tree identifier (must have data/captures/<tree_id>.mov)")
    parser.add_argument("--stride", type=int, default=3,
                        help="Frame extraction stride")
    parser.add_argument("--max-frames", type=int, default=120,
                        help="Max frames to extract")
    parser.add_argument("--quality", choices=["low", "medium", "high", "extreme"],
                        default="extreme",
                        help="COLMAP SIFT feature-count preset")
    parser.add_argument("--measured-trunk-height", type=float, default=None,
                        metavar="METRES",
                        help="Tape-measured trunk height in metres for scale calibration. "
                             "If omitted, predictions are reported in COLMAP units.")
    parser.add_argument("--hit-rate-threshold", type=float, default=0.6,
                        help="Reprojection filter: min hit rate to keep a point")
    parser.add_argument("--min-visible-frames", type=int, default=5,
                        help="Reprojection filter: min frame visibility to be eligible")
    parser.add_argument("--vertical-axis", type=int, choices=[0, 1, 2], default=2,
                        help="World axis that points up (depends on COLMAP orientation)")
    parser.add_argument("--graph-k", type=int, default=12,
                        help="kNN graph: neighbours per node")
    parser.add_argument("--graph-max-edge", type=float, default=2.0,
                        help="kNN graph: drop edges longer than this (COLMAP units)")
    parser.add_argument("--min-spur", type=float, default=0.30,
                        help="Skeleton spur pruning threshold (COLMAP units)")
    args = parser.parse_args()

    np.random.seed(131)

    tree_id = args.tree_id
    capture       = PROJECT_ROOT / "data" / "captures"     / f"{tree_id}.mov"
    intrinsics_p  = PROJECT_ROOT / "data" / "calibration"  / "intrinsics.npz"
    raw_frames    = PROJECT_ROOT / "data" / "frames"       / tree_id
    clahe_frames  = PROJECT_ROOT / "data" / "frames"       / f"{tree_id}_clahe"
    workspace     = PROJECT_ROOT / "outputs" / "reconstructions" / f"{tree_id}_colmap"
    raw_ply       = PROJECT_ROOT / "outputs" / "reconstructions" / f"{tree_id}_sparse.ply"
    filt_ply      = PROJECT_ROOT / "outputs" / "reconstructions" / f"{tree_id}_filtered.ply"
    graph_npz     = PROJECT_ROOT / "outputs" / "reconstructions" / f"{tree_id}_graph.npz"
    pred_csv      = PROJECT_ROOT / "outputs" / "metrics"   / f"{tree_id}_predictions.csv"
    fig_path_dir  = PROJECT_ROOT / "outputs" / "figures"

    if not intrinsics_p.exists():
        print(
            f"ERROR: intrinsics not found at {intrinsics_p}\n"
            "Run notebook 01 (camera calibration) once before using this script.",
            file=sys.stderr,
        )
        return 1

    print(f"Pipeline for: {tree_id}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Capture:      {capture}")

    t0 = time.time()

    banner("Stage 1: Frame extraction")
    n_frames = stage_extract_frames(capture, raw_frames, args.stride, args.max_frames)

    banner("Stage 2: CLAHE contrast preprocessing")
    stage_clahe(raw_frames, clahe_frames)

    banner(f"Stage 3: COLMAP ({args.quality} quality) — this may take a while")
    recon = stage_colmap(clahe_frames, workspace, args.quality)
    sfm.save_ply(raw_ply, recon.points_3d, recon.point_colors)
    print(f"Saved raw cloud → {raw_ply}")

    banner("Stages 4 + 5: Per-frame bark masks + reprojection filter")
    cloud, _ = stage_mask_and_filter(
        recon, clahe_frames,
        hit_rate_threshold=args.hit_rate_threshold,
        min_visible_frames=args.min_visible_frames,
    )
    sfm.save_ply(filt_ply, cloud)
    print(f"Saved filtered cloud → {filt_ply}")

    banner("Stages 6 + 7: RANSAC trunk + Dijkstra branch graph")
    fit, graph = stage_skeleton(
        cloud,
        vertical_axis=args.vertical_axis,
        k=args.graph_k,
        max_edge_length=args.graph_max_edge,
        min_spur_length=args.min_spur,
        collinear_tolerance_deg=12.0,
    )
    np.savez(
        graph_npz,
        nodes=graph.nodes,
        edges=np.array(graph.edges, dtype=np.int64),
        trunk_root=fit.root,
        trunk_direction=fit.direction,
        parents=np.array(list(graph.parent_of.items()), dtype=np.int64),
        distances=graph.distances,
    )
    print(f"Saved graph → {graph_npz}")

    banner("Stage 8: Per-branch measurements")
    preds, _scale, unit = stage_measurements(fit, graph, cloud, args.measured_trunk_height)
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["branch_id", f"attach_height_{unit}", "attach_angle_deg"])
        for p in preds:
            writer.writerow([p.branch_id, f"{p.attach_height_m:.3f}", f"{p.attach_angle_deg:.2f}"])
    print(f"Wrote {len(preds)} predicted primary branches → {pred_csv}")

    print()
    print(f"{'id':>3}  {'height':>12}  {'angle':>10}")
    for p in preds:
        print(f"{p.branch_id:>3}  {p.attach_height_m:>8.2f} {unit:<3}  {p.attach_angle_deg:>6.1f}°")

    banner("Stage 9: Final figure")
    fig_path_dir.mkdir(parents=True, exist_ok=True)
    fig = viz.plot_branch_graph(
        graph.nodes, graph.edges,
        trunk_axis=(fit.root, fit.direction),
        title=f"{tree_id} — {len(graph.nodes)} nodes, {len(preds)} primary branches",
    )
    out_fig = viz.save_fig(fig, f"pipeline_{tree_id}.png")
    print(f"Saved figure → {out_fig}")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Pipeline complete in {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"  Raw cloud:       {raw_ply}")
    print(f"  Filtered cloud:  {filt_ply}")
    print(f"  Branch graph:    {graph_npz}")
    print(f"  Predictions:     {pred_csv}")
    print(f"  Figure:          {out_fig}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
