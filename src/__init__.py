"""CS 131 final project — 3D tree reconstruction from skyward smartphone capture.

Stage modules (one per pipeline step):

    calibration   — intrinsics + frame extraction       (library-wrapped)
    features      — SIFT + matching + RANSAC F/E        (library-wrapped)
    two_view      — from-scratch E-matrix + DLT         (FROM SCRATCH)
    sfm           — COLMAP wrapper                      (library-wrapped)
    segmentation  — classical (scratch) + SAM (lib)
    filter_cloud  — reprojection-based cloud filter     (FROM SCRATCH)
    trunk         — RANSAC trunk-axis fit               (FROM SCRATCH)
    branch_graph  — Dijkstra branch skeleton            (FROM SCRATCH)
    evaluate      — metrics vs. ground truth            (library-wrapped)
    viz           — shared plotting helpers             (library-wrapped)
"""
