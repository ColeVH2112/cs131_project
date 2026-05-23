# 3D Reconstruction of Primary Tree Branch Structure from Skyward Multi-View Smartphone Capture

CS 131, Spring 2026 — Cole Van Hersett. The full proposal is in `CS131_proposal_draft.pdf`.

The system takes a short skyward smartphone orbit around a tree's base and outputs a
3D branch graph: trunk axis plus first-order branches with their attachment points,
angles, and growth directions.

## Pipeline

```
capture  →  calibration  →  SIFT + matching  →  reconstruction  →  segmentation  →  reproj-filter  →  trunk + branch graph  →  evaluation
                                                  ├─ from-scratch two-view       ├─ classical (from scratch)
                                                  └─ COLMAP multi-view           └─ SAM (library)
```

Each stage has a notebook (`notebooks/0X_*.ipynb`) and a module in `src/`.

## Layout

```
.
├── CS131_proposal_draft.pdf        ← project proposal (do not edit)
├── README.md
├── requirements.txt
├── data/                           (raw inputs — mostly gitignored)
│   ├── captures/                   .mov / .mp4 from phone
│   ├── frames/                     extracted frames per tree
│   ├── calibration/                checkerboard images + intrinsics.npz
│   └── ground_truth/               tape-measure CSVs per tree
├── src/                            (pipeline modules)
├── notebooks/                      (one notebook per pipeline stage)
└── outputs/
    ├── figures/                    paper-quality figures
    ├── reconstructions/            .ply clouds, poses, graphs (gitignored)
    └── metrics/                    per-tree CSVs
```

## Setup

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name cs131-tree --display-name "CS131 Tree"
```

### 2. COLMAP

The multi-view reconstruction notebook (`04_sfm_colmap.ipynb`) shells out to the
`colmap` CLI. Install it once and make sure `colmap` is on your `PATH`.

```bash
# macOS
brew install colmap

# Ubuntu
sudo apt install colmap

# Or build from source: https://colmap.github.io/install.html
```

### 3. SAM checkpoint

The segmentation notebook (`05_segmentation.ipynb`) uses the Segment Anything ViT-B
checkpoint. Download once into `checkpoints/`:

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
focal length) goes in `data/calibration/checkerboard.mov`.

## Running the pipeline

The notebooks are numbered in pipeline order:

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_calibration.ipynb` | Intrinsics from checkerboard; extract frames from capture |
| 02 | `02_features_matching.ipynb` | SIFT keypoints, ratio test, RANSAC verification |
| 03 | `03_two_view_scratch.ipynb` | **Milestone:** from-scratch normalized 8-point + DLT + cheirality, compared to cv2 reference |
| 04 | `04_sfm_colmap.ipynb` | **Milestone:** COLMAP multi-view reconstruction, sparse cloud of one tree |
| 05 | `05_segmentation.ipynb` | Classical (from scratch) vs SAM (library) sky-vs-structure masks, side by side |
| 06 | `06_filtering.ipynb` | Reproject every 3D point into every mask, keep consistent points (key visualization) |
| 07 | `07_skeleton.ipynb` | RANSAC trunk fit + Dijkstra branch graph from cleaned cloud |
| 08 | `08_evaluation.ipynb` | Metrics vs. tape-measure ground truth + three ablations |

Each notebook seeds `np.random.seed(131)` and saves at least one figure into
`outputs/figures/`.

## Milestone deliverable (5/22)

Two artifacts, both produced by notebooks above:

1. **Sparse 3D point cloud of one tree** showing primary branch structure — from `04_sfm_colmap.ipynb`, saved to `outputs/reconstructions/<tree_id>_sparse.ply` and rendered into `outputs/figures/milestone_cloud.png`.
2. **Working from-scratch two-view reconstruction** — from `03_two_view_scratch.ipynb`, with a side-by-side comparison against cv2's reference implementation saved to `outputs/figures/milestone_two_view.png`.

## What's implemented from scratch vs. library-wrapped

Per CS 131 convention, from-scratch code lives between `### YOUR CODE HERE` /
`### END YOUR CODE` markers.

**From scratch** (graders will inspect these):
- `src/two_view.py` — normalized 8-point, E-matrix decomposition, DLT triangulation, cheirality.
- `src/segmentation.py::classical_sky_mask` — brightness threshold + Canny edges.
- `src/filter_cloud.py` — reprojection-based cloud filtering (core novelty).
- `src/trunk.py` — RANSAC trunk-axis fit.
- `src/branch_graph.py` — Dijkstra shortest-path tree, spur pruning, collinear merging.

**Library-wrapped** (provided as utilities):
- `src/calibration.py` — cv2 checkerboard calibration and frame extraction.
- `src/features.py` — SIFT detection, Lowe's ratio test, cv2 RANSAC F/E.
- `src/sfm.py` — COLMAP CLI wrapper.
- `src/segmentation.py::sam_sky_mask` — SAM predictor wrapper.
- `src/evaluate.py`, `src/viz.py` — metrics + plotting helpers.
