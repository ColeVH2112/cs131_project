# 3D Reconstruction of Primary Tree Branch Structure from Skyward Multi-View Smartphone Capture

**Cole Van Hersett** — CS 131, Spring 2026 — Solo Project

## Problem Statement

Recovering the three-dimensional structure of a tree — the geometry of its primary branches, not merely its height or trunk — is a long-standing problem in computer vision with direct applications to forestry, arboriculture, and hazard assessment. Existing approaches rely predominantly on terrestrial LiDAR or specialized multi-camera rigs; image-only methods using consumer hardware remain limited because leaf occlusion hides branch structure from typical outside-the-tree viewpoints.

This project investigates a capture strategy designed around tree-specific occlusion geometry: **skyward orbital capture from the base of the trunk**. The camera moves in a short arc around the trunk while pointed upward, looking through the canopy from below. This exploits two structural facts: (1) primary branches near the trunk lie above most of the leaf mass and are largely unoccluded from below, and (2) branches silhouetted against the sky form high-contrast features favorable for both correspondence matching and segmentation.

The contribution is a pipeline that takes such a video and outputs a **3D branch graph** rooted at the trunk base — a structured representation of primary branches with their attachment points, attachment angles, and directions — together with empirical accuracy characterization across species and lighting conditions. The pipeline spans all three course units: low-level vision for filters, edges, and classical segmentation; geometry for multi-view reconstruction; and ML for CV for a learned segmentation comparison. The algorithmic core adapts the graph-based skeleton extraction of Livny et al. [1] — originally designed for dense LiDAR point clouds — to the sparser, noisier point clouds produced by image-only reconstruction of thin structures. This adaptation is itself a contribution.

Output is scoped to *primary* branch structure (trunk and first-order branches, proximal portions). Fine branches and twigs are out of scope and are treated as characterized limitations rather than targets.

## Proposed Methodology

The pipeline has five stages. Implementation is in Python, using OpenCV, NumPy, COLMAP, Open3D, and the Segment Anything (SAM) pretrained checkpoint.

1. **Calibration and capture** (Lecture 6: Calibration). Camera intrinsics are estimated once via checkerboard. Capture protocol: 15–25 second video, ~2-meter orbit radius around the trunk, camera tilted ~30° above horizontal, with deliberate translation between frames to avoid the pure-rotation degeneracy that breaks two-view geometry.

2. **Feature detection and correspondence** (Lecture 4: Local Features; Lecture 2: Filters). SIFT keypoints [2] are extracted across frames; branches against sky yield strong corner-like features at junctions and silhouette transitions. Correspondences are matched with Lowe's ratio test and geometrically verified via RANSAC.

3. **Multi-view reconstruction** (Lectures 8–9: Multi-View Geometry). Essential matrix estimation via the 8-point algorithm with RANSAC is implemented from scratch on a two-view subset, satisfying course-aligned implementation expectations. Full multi-view reconstruction for camera pose recovery and sparse triangulation is performed via COLMAP [3]. The from-scratch two-view component additionally serves as a benchmark against the full pipeline.

4. **Branch segmentation — classical and learned in parallel** (Lectures 2–3: Filters and Edges; Lectures 11–13: ML for CV). Sky-vs-structure segmentation is performed two ways: (a) a classical pipeline using brightness thresholding and Canny edge detection, and (b) a learned pipeline using SAM [4] with a trunk-base point prompt. Both produce per-frame structure masks and are compared as an ablation. Segmentation additionally enables silhouette-based reasoning that complements point-cloud SfM: pixels labeled "structure" define 3D rays under the recovered camera poses, and branch geometry is reinforced where these rays intersect consistently across views (Lecture 7: Single View Metrology). This step is the primary mechanism for recovering thin structures that sparse SfM misses.

5. **Trunk-rooted skeleton extraction.** Adapted from Livny et al. [1], whose original work uses multi-root Dijkstra and global optimization on dense LiDAR scans. Here the trunk axis is fit first via RANSAC on the dense vertical structure near the camera origin, fixing a single known root. A weighted graph is constructed over the structure-labeled 3D points, with edges between spatial neighbors weighted by Euclidean distance, and a shortest-path tree rooted at the trunk base is computed via Dijkstra's algorithm. Short spurious branches are pruned and collinear segments are merged via 3D RANSAC line fitting (a natural 3D extension of Lecture 3's Hough-line ideas). The final output is a directed branch graph in which each primary branch is characterized by attachment point, attachment angle, direction vector, and proximal length.

**Evaluation.** Dataset: 8–12 trees on Stanford campus, prioritizing species with clear primary branching (oaks, sycamores, maples). Ground truth is collected on a subset via tape measure (trunk diameter, attachment heights) and protractor (attachment angles). Quantitative metrics: primary branch count recall, attachment angle mean absolute error, and attachment height error. Built-in ablations compare classical vs learned segmentation, two-view vs full multi-view reconstruction, and the contribution of silhouette-based reinforcement. Qualitative analysis covers failure modes by species, lighting condition, and leaf density.

## Feasibility and Timeline

- **Week 1 (5/16–5/22):** Calibration; capture protocol pilot on 2–3 trees; SIFT correspondence; two-view essential matrix estimation and triangulation implemented from scratch. *Milestone deliverable: sparse 3D point cloud of one tree with visible primary branch structure, plus working from-scratch two-view reconstruction on a benchmark pair.*
- **Week 2 (5/23–5/29):** Classical and SAM-based segmentation pipelines; full COLMAP integration; trunk axis extraction; initial skeleton implementation; dataset expanded to 8+ trees.
- **Week 3 (5/30–6/2):** Branch graph pruning and segment merging; silhouette-based reinforcement; ground truth collection; quantitative evaluation; demo day slides (due 6/2).
- **Week 4 (6/3–6/6):** Ablations across segmentation methods, view counts, and reinforcement; failure mode analysis; final report.

**Primary risks and mitigations.** (1) *Rotational degeneracy in SfM:* enforced via capture protocol with explicit translation; verified per-capture by checking recovered baseline length against a minimum threshold. (2) *Thin-structure reconstruction failure:* skyward capture, sky-segmentation prior, and silhouette-based reinforcement specifically target this; remaining failures are characterized as limitations rather than treated as blockers. (3) *Leaf density variation in May:* species selection is biased toward early-leafing trees with sparse canopies, and leaf density itself becomes an experimental variable. (4) *Skeleton hyperparameter sensitivity:* tuned on a held-out subset and sensitivity reported in the final analysis.

## References

[1] Livny, Y., Yan, F., Olson, M., Chen, B., Zhang, H., El-Sana, J. *Automatic Reconstruction of Tree Skeletal Structures from Point Clouds.* ACM Transactions on Graphics (SIGGRAPH Asia), 29(6):151, 2010.
[2] Lowe, D. *Distinctive Image Features from Scale-Invariant Keypoints.* International Journal of Computer Vision, 60(2):91–110, 2004.
[3] Schönberger, J., Frahm, J.-M. *Structure-from-Motion Revisited.* CVPR, 2016.
[4] Kirillov, A., et al. *Segment Anything.* ICCV, 2023.
