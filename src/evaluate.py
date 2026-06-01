"""Evaluation metrics for the recovered branch graph vs. ground truth.

Ground truth is collected with tape measure and protractor on a subset of trees:
for each tree we record, per primary branch, the height above the trunk base of
the attachment point and the angle of the branch off the trunk axis. CSVs live
in `data/ground_truth/<tree_id>.csv` with columns:

    branch_id,attach_height_m,attach_angle_deg

The proposal lists three headline metrics:

    1. primary branch count recall   — fraction of GT branches matched
    2. attachment angle mean absolute error (deg)
    3. attachment height error (m)

Plus three ablations: classical vs. learned segmentation, two-view vs. multi-view
reconstruction, segmentation-based filtering on vs. off. Each ablation just
re-runs `evaluate_tree` with different upstream settings and stacks the rows
into `outputs/metrics/<run>.csv`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class GTBranch:
    """One ground-truth primary branch."""
    branch_id: int
    attach_height_m: float
    attach_angle_deg: float


@dataclass
class PredBranch:
    """One predicted primary branch from the branch graph."""
    branch_id: int
    attach_height_m: float
    attach_angle_deg: float


@dataclass
class TreeMetrics:
    """Per-tree evaluation result.

    `recall` = matched / n_gt  (did we find the real branches?).
    `precision` = matched / n_pred  (how many predicted branches were real?).
    Reporting both is important here: Dijkstra over a noisy cloud tends to
    over-produce "primary branches", so a high recall can hide heavy
    over-segmentation. n_pred vs n_gt makes that visible.
    """
    tree_id: str
    n_gt: int
    n_pred: int
    n_matched: int
    recall: float
    precision: float
    angle_mae_deg: float
    height_mae_m: float


def load_ground_truth(csv_path: str | Path) -> list[GTBranch]:
    """Load tape-measure ground-truth for one tree.

    Blank lines and `#`-comment lines are ignored, so the CSV can carry a
    measurement-convention header (see data/ground_truth/tree_5.csv).
    """
    out: list[GTBranch] = []
    with open(csv_path) as f:
        rows = (ln for ln in f if ln.strip() and not ln.lstrip().startswith("#"))
        reader = csv.DictReader(rows)
        for row in reader:
            out.append(GTBranch(
                branch_id=int(row["branch_id"]),
                attach_height_m=float(row["attach_height_m"]),
                attach_angle_deg=float(row["attach_angle_deg"]),
            ))
    return out


def predicted_branches_from_graph(
    nodes: np.ndarray,
    edges: list[tuple[int, int]],
    trunk_root: np.ndarray,
    trunk_axis: np.ndarray,
    parent_of: dict[int, int],
    on_trunk_thresh: float = 0.05,
) -> list[PredBranch]:
    """Walk the branch graph and emit one PredBranch per primary branch.

    A "primary branch" is an edge that leaves the trunk: its parent endpoint
    lies on the trunk and its child does not. Attachment height = signed distance
    along the trunk axis from the root. Attachment angle = angle between the
    branch's initial segment direction and the trunk axis.

    NOTE ON UNITS: this runs on `nodes` in whatever units the cloud is in. In the
    production pipeline (scripts/run_pipeline.py) the metric scale is applied
    *after* this call, so here `nodes`, `trunk_root`, and `on_trunk_thresh` are
    all in arbitrary COLMAP units — NOT metres. `on_trunk_thresh` therefore has
    to be tuned per-reconstruction (this is one reason a per-tree set of graph
    thresholds tuned on tree_5 does not transfer unchanged to other trees).

    Args:
        nodes: (N, 3) skeleton node positions (COLMAP units unless pre-scaled).
        edges: list of (parent_idx, child_idx) skeleton edges.
        trunk_root: (3,) coords of the trunk base, same units as `nodes`.
        trunk_axis: (3,) unit vector along the trunk (pointing up the tree).
        parent_of: child → parent map produced by Dijkstra in branch_graph.
        on_trunk_thresh: max perpendicular distance (in `nodes` units) for a
            node to count as lying on the trunk axis.

    Returns:
        List of PredBranch entries, one per detected primary branch.
    """
    axis = trunk_axis / (np.linalg.norm(trunk_axis) + 1e-12)
    out: list[PredBranch] = []
    bid = 0
    # An edge is a "primary" branch if it transitions off-trunk. We mark a
    # node as on-trunk if it's within `on_trunk_thresh` of the trunk line.
    # (branch_graph already prunes spurs / merges segments, so this is cheap.)
    on_trunk = _within_distance_of_line(nodes, trunk_root, axis, on_trunk_thresh)

    for parent_idx, child_idx in edges:
        if on_trunk[parent_idx] and not on_trunk[child_idx]:
            p = nodes[parent_idx]; c = nodes[child_idx]
            h = float(np.dot(p - trunk_root, axis))
            dirn = (c - p) / (np.linalg.norm(c - p) + 1e-12)
            angle = float(np.degrees(np.arccos(np.clip(np.dot(dirn, axis), -1.0, 1.0))))
            out.append(PredBranch(branch_id=bid, attach_height_m=h, attach_angle_deg=angle))
            bid += 1
    return out


def _within_distance_of_line(
    points: np.ndarray, p0: np.ndarray, d: np.ndarray, thresh: float,
) -> np.ndarray:
    """Boolean mask: True where point lies within `thresh` of an infinite line."""
    v = points - p0[None, :]
    proj = (v @ d)[:, None] * d[None, :]
    perp = v - proj
    return np.linalg.norm(perp, axis=1) < thresh


def match_branches(
    preds: list[PredBranch], gts: list[GTBranch],
    height_tolerance_m: float = 0.30,
) -> list[tuple[int, int]]:
    """Greedy match: for each GT, pick the closest-in-height unmatched prediction.

    Returns a list of (gt_idx, pred_idx) pairs whose height difference is
    within `height_tolerance_m`. Unmatched GTs count as misses; unmatched
    predictions count as false positives.
    """
    gts_sorted = sorted(enumerate(gts), key=lambda kv: kv[1].attach_height_m)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for gi, gt in gts_sorted:
        best = None
        best_dh = height_tolerance_m
        for pi, pred in enumerate(preds):
            if pi in used:
                continue
            dh = abs(pred.attach_height_m - gt.attach_height_m)
            if dh < best_dh:
                best_dh = dh; best = pi
        if best is not None:
            pairs.append((gi, best))
            used.add(best)
    return pairs


def evaluate_tree(
    tree_id: str,
    preds: list[PredBranch],
    gts: list[GTBranch],
    height_tolerance_m: float = 0.30,
) -> TreeMetrics:
    """Compute recall, attachment angle MAE, attachment height MAE for one tree."""
    pairs = match_branches(preds, gts, height_tolerance_m)
    if not pairs:
        return TreeMetrics(
            tree_id=tree_id, n_gt=len(gts), n_pred=len(preds),
            n_matched=0, recall=0.0, precision=0.0,
            angle_mae_deg=float("nan"), height_mae_m=float("nan"),
        )
    angle_errs = [abs(preds[pi].attach_angle_deg - gts[gi].attach_angle_deg) for gi, pi in pairs]
    height_errs = [abs(preds[pi].attach_height_m - gts[gi].attach_height_m) for gi, pi in pairs]
    return TreeMetrics(
        tree_id=tree_id,
        n_gt=len(gts), n_pred=len(preds), n_matched=len(pairs),
        recall=len(pairs) / max(1, len(gts)),
        precision=len(pairs) / max(1, len(preds)),
        angle_mae_deg=float(np.mean(angle_errs)),
        height_mae_m=float(np.mean(height_errs)),
    )


def write_metrics_csv(rows: list[TreeMetrics], path: str | Path) -> None:
    """Write a list of TreeMetrics rows to outputs/metrics/<run>.csv."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
