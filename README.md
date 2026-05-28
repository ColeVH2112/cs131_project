# 3D Reconstruction of Primary Tree Branch Structure from Skyward Multi-View Smartphone Capture

CS 131, Spring 2026 — Cole Van Hersett. The full proposal is in `CS131_proposal_draft.pdf`.

The system takes a short skyward smartphone orbit around a tree's base and outputs a
3D branch graph: trunk axis plus first-order branches with their attachment heights,
angles, and growth directions.

> **Status:** all five from-scratch modules are implemented (two-view geometry,
> classical segmentation, reprojection filter, RANSAC trunk fit, Dijkstra branch
> graph). End-to-end pipeline runs via one CLI command (see [Quick start](#quick-start-cli)).
> Remaining work for the final report is ground-truth measurement + ablation runs.

## Quick start (CLI)

After [setup](#setup), running the full pipeline on one tree is a single command:

```bash
python scripts/run_pipeline.py tree_5 --measured-trunk-height 3.5
```

That command:

```
data/captures/tree_5.mov
   → frame extraction       (skipped if cached)
   → CLAHE preprocessing    (skipped if cached)
   → COLMAP multi-view SfM  (extreme quality, sequential matcher)
   → LAB bark masks         (per-frame, via src/segmentation.py)
   → reprojection filter    (the proposal's core novelty)
   → statistical denoise
   → RANSAC trunk axis      (with vertical prior)
   → Dijkstra branch graph  (kNN + shortest-path tree + spur prune + collinear merge)
   → metric scaling         (from the tape-measured trunk height)
   → final figure
```

Outputs land in:

| Path | What |
|---|---|
| `outputs/reconstructions/<tree_id>_sparse.ply` | raw COLMAP cloud |
| `outputs/reconstructions/<tree_id>_filtered.ply` | after reprojection filter |
| `outputs/reconstructions/<tree_id>_graph.npz` | branch graph (nodes, edges, parents, trunk) |
| `outputs/metrics/<tree_id>_predictions.csv` | per-branch height + angle |
| `outputs/figures/pipeline_<tree_id>.png` | annotated branch-graph figure |

See `python scripts/run_pipeline.py --help` for tuning knobs (quality preset,
reprojection-filter thresholds, trunk vertical-axis selection, kNN graph
parameters).

## Pipeline

```
capture → calibration → SIFT + matching → reconstruction → segmentation → reproj-filter → trunk + branch graph → evaluation
                                          ├─ from-scratch two-view      ├─ classical (from scratch)
                                          └─ COLMAP multi-view          └─ LAB bark mask (from scratch)
```

Each stage has a notebook (`notebooks/0X_*.ipynb`) for interactive iteration and a
module in `src/` for code reuse. The CLI runner above bypasses the notebooks and
calls the `src/` modules directly.

**Note on segmentation pipeline placement.** The proposal called for sky-vs-structure
segmentation as a pre-filter on COLMAP feature extraction. An empirical Week-3
ablation found this *reduced* COLMAP frame registration from 99% to 54% — the
per-frame masks varied enough across the orbit that SIFT features near mask
boundaries became inconsistent, hurting multi-view matching. The masks are now
applied **post-hoc** in the reprojection filter (notebook 06 / pipeline stage 4),
which is the proposal's intended use of them and works as expected.

## Layout

```
.
├── CS131_proposal_draft.pdf        ← project proposal (do not edit)
├── README.md
├── requirements.txt
├── scripts/
│   └── run_pipeline.py             ← single-command end-to-end runner
├── src/                            (pipeline modules — five from-scratch + four library-wrapped)
├── notebooks/                      (one notebook per pipeline stage, for interactive use)
├── data/                           (raw inputs — gitignored)
│   ├── captures/                   .mov / .mp4 from phone
│   ├── frames/                     extracted frames per tree
│   ├── calibration/                checkerboard images + intrinsics.npz
│   └── ground_truth/               tape-measure CSVs per tree
└── outputs/
    ├── figures/                    paper-quality figures
    ├── reconstructions/            .ply clouds, poses, graphs (gitignored)
    └── metrics/                    per-tree predictions + per-run ablation CSVs
```

## Setup

### 1. Python environment

The project is tested with **Python 3.12**. Avoid 3.14 (some downstream packages
don't have wheels yet).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name cs131-tree --display-name "CS131 Tree"
```

### 2. COLMAP

Both the notebooks and `scripts/run_pipeline.py` shell out to the `colmap` CLI.
Install it once and make sure `colmap` is on your `PATH`.

```bash
# macOS
brew install colmap

# Ubuntu
sudo apt install colmap

# Or build from source: https://colmap.github.io/install.html
```

The macOS Homebrew COLMAP 4.0.4 build is compiled without the CUDA SIFT GPU
flag — the `src/sfm.py` wrapper drops `--SiftExtraction.use_gpu` and friends
so it works on that build out of the box.

### 3. SAM checkpoint (optional)

The original proposal scoped SAM (Segment Anything) as the learned-segmentation
alternative for the classical-vs-learned ablation. The current pipeline uses
a **from-scratch LAB bark mask** (`src/segmentation.py::bark_color_mask_lab`)
instead of SAM, which removes the checkpoint dependency. The `SamSegmenter`
class in `src/segmentation.py` remains available if you want to reproduce the
proposal's intended ablation; if so, download the ViT-B checkpoint:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

The checkpoint is gitignored.

## Capture protocol

The capture is the most original part of the method; follow it carefully.

1. Pick a tree with clear primary branching (oak, sycamore, maple work well).
   Stand directly under the canopy, near the trunk base.
2. Hold the phone overhead, camera **pointed upward** and **tilted ~30° off vertical**
   so branches are silhouetted against the sky.
3. Walk a slow **~2 m radius orbit** around the trunk while continuously recording.
   **Translate, do not just rotate** — pure rotation makes two-view geometry degenerate.
4. Total clip length **15–25 seconds**, single take, 30 fps or higher.
5. Save as `data/captures/<tree_id>.mov` (e.g. `tree_oak_01.mov`).

The calibration video (a checkerboard waved in front of the camera, same device and
focal length) goes in `data/calibration/checkerboard.mov`. Run notebook 01 once to
extract intrinsics into `data/calibration/intrinsics.npz`; the CLI runner reuses
those for every subsequent tree.

## Running the pipeline

### Option A — single CLI command (recommended)

See [Quick start](#quick-start-cli) above. This is the production path: one
command, no Jupyter, ~25 minutes per tree at extreme quality.

### Option B — interactive notebooks (per-stage inspection)

For tuning or one-off inspection, the notebooks expose each stage:

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_calibration.ipynb` | Intrinsics from checkerboard; extract frames from capture |
| 02 | `02_features_matching.ipynb` | SIFT keypoints, ratio test, RANSAC verification |
| 03 | `03_two_view_scratch.ipynb` | From-scratch normalized 8-point + DLT + cheirality, compared to cv2 reference |
| 04 | `04_sfm_colmap.ipynb` | COLMAP multi-view reconstruction, sparse cloud of one tree |
| 05 | `05_segmentation.ipynb` | Classical (from scratch) vs SAM (library) sky-vs-structure masks |
| 06 | `06_filtering.ipynb` | Reproject every 3D point into every mask; keep consistent points (key visualization) |
| 07 | `07_skeleton.ipynb` | RANSAC trunk fit + Dijkstra branch graph from cleaned cloud |
| 08 | `08_evaluation.ipynb` | Metrics vs. tape-measure ground truth + three ablations |

Each notebook seeds `np.random.seed(131)` and saves at least one figure into
`outputs/figures/`.

## Milestone deliverable (5/22)

Two artifacts, both produced by notebooks above:

1. **Sparse 3D point cloud of one tree** showing primary branch structure — from `04_sfm_colmap.ipynb`, saved to `outputs/reconstructions/<tree_id>_sparse.ply` and rendered into `outputs/figures/milestone_cloud.png`.
2. **Working from-scratch two-view reconstruction** — from `03_two_view_scratch.ipynb`, with a side-by-side comparison against cv2's reference implementation saved to `outputs/figures/milestone_two_view.png`. The from-scratch implementation agrees with OpenCV's reference to within **0.058° rotation, 0.031° translation direction** on real SIFT correspondences.

## What's implemented from scratch vs. library-wrapped

Per CS 131 convention: `### YOUR CODE HERE` / `### END YOUR CODE` markers. **All
five from-scratch modules are now complete (zero unimplemented stubs).**

- `src/two_view.py` — Hartley normalisation, normalised 8-point, F→E conversion, essential decomposition, DLT triangulation, cheirality check.
- `src/segmentation.py::classical_sky_mask` — brightness threshold + Canny edge + morphological closing.
- `src/segmentation.py::bark_color_mask_lab` — LAB chrominance + Mahalanobis distance + vertical spatial prior + gradient-density texture rescue (replaces SAM as the segmenter in the current pipeline).
- `src/filter_cloud.py` — reprojection-based cloud filtering (the proposal's core novelty) + kNN statistical outlier removal + voxel-grid downsampling.
- `src/trunk.py` — point-to-line distance, PCA line fit, RANSAC trunk-axis fit with a near-vertical prior + PCA refit + sign canonicalisation.
- `src/branch_graph.py` — kNN graph construction, heap-based Dijkstra shortest-path tree, iterative spur pruning, collinear segment merging.

**Library-wrapped** (provided as utilities):

- `src/calibration.py` — cv2 checkerboard calibration (Zhang's method) and frame extraction.
- `src/features.py` — SIFT detection, Lowe's ratio test, cv2 RANSAC F/E matrix verification.
- `src/sfm.py` — COLMAP CLI wrapper (lower-level `feature_extractor` → `sequential_matcher` → `mapper` pipeline with optional `--ImageReader.mask_path`).
- `src/segmentation.py::sam_sky_mask` — SAM predictor wrapper (optional).
- `src/evaluate.py`, `src/viz.py` — metrics + plotting helpers.

## AI assistance disclosure

Scaffolding code — module layout, library wrappers (the COLMAP CLI interface, the
COLMAP binary-format readers, the `.ply` writer, plotting helpers), notebook
narrative skeletons, the CLI argument parser in `scripts/run_pipeline.py`, and the
docstring-driven function signatures for the from-scratch modules — was produced
with AI assistance. All from-scratch algorithmic implementations marked with
`### YOUR CODE HERE` / `### END YOUR CODE` blocks were developed interactively
with AI assistance during pair-programming style iteration, then committed
under those markers.

The CLAHE + LAB bark-mask preprocessing pipeline (added during Week 3 after the
empirical finding that pre-COLMAP masking hurt registration) was developed
interactively with AI assistance as well; the mask itself lives in
`src/segmentation.py::bark_color_mask_lab`, and the pipeline orchestration
lives in `scripts/run_pipeline.py` and `notebooks/04`, `06`, `08`.
