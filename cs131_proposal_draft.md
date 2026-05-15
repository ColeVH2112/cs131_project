# 3D Reconstruction of Primary Tree Branch Structure from Skyward Multi-View Smartphone Capture

**Cole Van Hersett** — CS 131, Spring 2026 — Solo Project

## Problem and Motivation

Recovering the 3D structure of a tree — the geometry of its primary branches, not just its height or trunk — is an open problem in computer vision with real applications in forestry, arboriculture, and hazard assessment. Knowing where a tree's main branches attach to the trunk and which direction they grow is what drives felling planning, rigging, pruning, and biomass estimation. Existing solutions rely almost entirely on terrestrial LiDAR or specialized camera rigs that cost thousands of dollars; image-only methods using consumer phones remain limited because leaf cover hides branch structure from typical outside-the-tree viewpoints.

I chose this problem because: (1) I run a tree removal service, so the gap between what a phone sees and what an arborist needs is something I have felt firsthand; (2) it draws on every major unit of CS 131 — low-level vision, geometry, and ML for CV; and (3) the approach is original in the image-based setting — most published image-only tree work targets graphics realism rather than measurement, or assumes leaf-off conditions unavailable in May at Stanford.

The originality of the method comes from a capture strategy designed around tree-specific occlusion geometry: **skyward orbital capture from the base of the trunk**. The camera moves in a short arc around the trunk while pointed upward, looking through the canopy from below. This exploits two facts about trees that, to my knowledge, no image-based method explicitly leverages: primary branches near the trunk lie above most of the leaf mass, and branches silhouetted against the sky form high-contrast features ideal for matching and segmentation. Output is scoped to primary branch structure (trunk and first-order branches, proximal portions); finer twigs are out of scope.

## System Overview

The system takes a short smartphone video of someone walking around a tree's base while pointing the camera up, and outputs a **3D branch graph** rooted at the trunk: a structured representation of primary branches with their attachment points, angles, and growth directions. End to end, the pipeline (1) detects and matches SIFT keypoints across frames, (2) recovers 3D camera poses and a sparse point cloud via multi-view geometry, (3) uses per-frame sky-vs-structure segmentation to filter the cloud down to points actually on the tree, (4) fits the trunk as a 3D line, and (5) builds the branch graph by running Dijkstra's shortest-path algorithm from the trunk base outward through the cleaned points. The conceptual fit is direct: real trees are tree-shaped graphs (the trunk forks into branches, which fork further, with no rejoining), so a shortest-path tree rooted at the trunk physically corresponds to the tree's own branching structure. Measurements are read off the final graph.

## Methodology and Course Connections

Implementation in Python using OpenCV, NumPy, COLMAP, Open3D, and the Segment Anything (SAM) checkpoint. The five stages each draw on specific CS 131 material:

**(1) Calibration and capture** (Lec. 6): camera intrinsics via checkerboard; 15–25 sec video, ~2 m orbit, camera tilted ~30° above horizontal, with deliberate translation between frames to avoid pure-rotation degeneracy.

**(2) Feature matching** (Lec. 2, 4): SIFT keypoints [2] matched across frames with Lowe's ratio test and RANSAC verification.

**(3) Multi-view reconstruction** (Lec. 8–9): essential matrix and triangulation implemented from scratch on a two-view subset; full pose recovery and sparse cloud via COLMAP [3].

**(4) Sky-vs-structure segmentation** (Lec. 2–3, 11–13), done two ways and compared: classical (brightness threshold + Canny edges) and learned (SAM [4] with a trunk-base prompt). The masks then filter the 3D cloud by reprojecting each point into every frame and keeping only those that consistently land in "structure" pixels — this is the main mechanism for cleaning the cloud and recovering thin branches that raw SfM misses.

**(5) Trunk fitting and branch graph extraction**, adapted from Livny et al. [1]: trunk axis fit by RANSAC, weighted graph built over the cleaned points, shortest-path tree from the trunk root computed via Dijkstra, short spurious branches pruned, collinear segments merged via 3D RANSAC line fitting (a natural extension of Lec. 3's Hough-line ideas).

**Evaluation:** 8–12 trees on Stanford campus, prioritizing species with clear primary branching (oaks, sycamores, maples). Ground truth on a subset via tape measure and protractor. Metrics: primary branch count recall, attachment angle mean absolute error, attachment height error. Built-in ablations: classical vs. learned segmentation, two-view vs. full multi-view reconstruction, segmentation-based filtering on vs. off.

## Feasibility and Timeline

The scope is tight but realistic for a solo project. Weeks 1–2: build the geometric pipeline (calibration, SIFT, from-scratch two-view reconstruction, COLMAP integration, initial captures). Week 3: segmentation comparison, skeleton extraction, and ground-truth collection. Week 4: ablations, failure-mode analysis, and final report. Milestone deliverable (5/22): a sparse 3D point cloud of one tree with visible primary branch structure, plus a working from-scratch two-view reconstruction. Main risks: (i) rotational degeneracy in SfM, addressed by capture protocol and per-capture baseline checks; (ii) thin-structure reconstruction failure, addressed by skyward capture and segmentation-based filtering; (iii) leaf density variation in May, addressed by species selection and treated as an experimental variable.

## References

[1] Livny, Y. et al. *Automatic Reconstruction of Tree Skeletal Structures from Point Clouds.* ACM TOG (SIGGRAPH Asia), 2010.
[2] Lowe, D. *Distinctive Image Features from Scale-Invariant Keypoints.* IJCV, 2004.
[3] Schönberger, J., Frahm, J.-M. *Structure-from-Motion Revisited.* CVPR, 2016.
[4] Kirillov, A. et al. *Segment Anything.* ICCV, 2023.
