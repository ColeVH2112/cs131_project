"""Stage 4 — sky-vs-structure segmentation, two ways.

The skyward capture geometry means that "structure" (trunk + branches) appears
as dark, edge-rich pixels silhouetted against bright, edge-poor sky. We
exploit that two ways and compare them as an ablation:

    classical_sky_mask  — FROM SCRATCH: brightness threshold + Canny edges
    sam_sky_mask        — LIBRARY: Segment Anything prompted with the trunk base

**Mask convention** (shared by both functions, enforced here):

    mask.dtype == bool
    mask.shape == (H, W)
    mask[y, x] == True   →  pixel is STRUCTURE (tree / trunk / branch)
    mask[y, x] == False  →  pixel is SKY / background

The reprojection filter in `filter_cloud.py` counts how often each 3D point
lands in `mask == True` across all frames, so swapping classical for SAM is a
one-line change at the call site. Do not flip the convention.

Both functions take the same signature `(image_bgr, **kwargs) -> mask`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Classical sky-vs-structure  —  FROM SCRATCH
# ---------------------------------------------------------------------------

def classical_sky_mask(
    image_bgr: np.ndarray,
    brightness_thresh: int = 200,
    canny_low: int = 50,
    canny_high: int = 150,
    morph_kernel: int = 5,
) -> np.ndarray:
    """Classical sky-vs-structure segmentation by brightness + edges.

    Intuition: against a bright overcast sky (the typical skyward-capture
    appearance), trunk and branches are darker than the sky AND surrounded by
    strong edges; sky pixels are bright AND smooth. We combine:

        bright_and_smooth = (gray > brightness_thresh) & (no nearby Canny edge)

    then invert to get a structure mask. A morphological closing fills small
    holes inside branches.

    Args:
        image_bgr: (H, W, 3) BGR image.
        brightness_thresh: pixels brighter than this in grayscale are candidate sky.
        canny_low, canny_high: Canny hysteresis thresholds.
        morph_kernel: kernel size for the structure-side closing operation.

    Returns:
        (H, W) boolean mask. True = structure, False = sky.
    """
    ### YOUR CODE HERE
    # Grayscale: brightness is the cheap proxy for "is this pixel sky-bright".
    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3 else image_bgr
    )

    # Stage 1: brightness threshold. Pixels brighter than this are sky candidates.
    bright = gray > brightness_thresh

    # Stage 2: Canny edges. Branch silhouettes against the sky produce strong
    # local gradients; the sky interior does not. We dilate the edge map so
    # "near an edge" is a small neighbourhood, not just the exact edge pixels —
    # this lets us label trunk-edge pixels (which are bright at the boundary
    # but adjacent to a strong silhouette gradient) as structure rather than sky.
    edges = cv2.Canny(gray, canny_low, canny_high)
    near_edge = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    ) > 0

    # Sky = bright AND not near any edge. Structure is its complement.
    sky = bright & ~near_edge
    structure = ~sky

    # Stage 3: morphological closing fills small interior holes inside branches
    # — places where a bright sky-coloured pixel survived the brightness gate
    # because it didn't happen to be near a detected Canny edge.
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    structure_u8 = cv2.morphologyEx(
        structure.astype(np.uint8), cv2.MORPH_CLOSE, close,
    )
    return structure_u8.astype(bool)
    ### END YOUR CODE


# ---------------------------------------------------------------------------
# SAM sky-vs-structure  —  LIBRARY-WRAPPED
# ---------------------------------------------------------------------------

class SamSegmenter:
    """Lazy-loaded Segment Anything predictor.

    SAM weights are big (~360 MB for ViT-B), so we instantiate the predictor
    once and reuse it across frames. The default prompt strategy is:

        - A point prompt at the bottom-centre of the image (where the trunk
          base typically sits in skyward captures).
        - Take the highest-scoring returned mask; that mask is structure.

    If a per-frame prompt list is supplied, those override the default.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/sam_vit_b_01ec64.pth",
        model_type: str = "vit_b",
        device: str | None = None,
    ):
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError as e:
            raise ImportError(
                "segment-anything is not installed. Run "
                "`pip install -r requirements.txt`."
            ) from e

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {ckpt}. Download it (see README) "
                "then re-run."
            )

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_type = model_type
        self.device = device
        sam = sam_model_registry[model_type](checkpoint=str(ckpt))
        sam.to(device=device)
        self._predictor = SamPredictor(sam)

    def __call__(
        self,
        image_bgr: np.ndarray,
        prompt_points: np.ndarray | None = None,
        prompt_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        return sam_sky_mask(
            image_bgr, predictor=self._predictor,
            prompt_points=prompt_points, prompt_labels=prompt_labels,
        )


def sam_sky_mask(
    image_bgr: np.ndarray,
    predictor=None,
    prompt_points: np.ndarray | None = None,
    prompt_labels: np.ndarray | None = None,
) -> np.ndarray:
    """SAM-based structure mask with a trunk-base prompt.

    Args:
        image_bgr: (H, W, 3) BGR image.
        predictor: a `segment_anything.SamPredictor` instance. If None, this
            function creates one (slow — prefer reusing a `SamSegmenter`).
        prompt_points: optional (K, 2) array of (x, y) prompt locations. If
            None, we use a single point at the bottom-centre of the image.
        prompt_labels: optional (K,) array; 1 = foreground (structure),
            0 = background. Defaults to all-foreground.

    Returns:
        (H, W) boolean mask. True = structure, False = sky.  Same convention as
        classical_sky_mask so the two are drop-in interchangeable.
    """
    if predictor is None:
        # Build a one-shot predictor. Notebook callers should pass one in.
        predictor = SamSegmenter()._predictor

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    predictor.set_image(image_rgb)

    if prompt_points is None:
        # Trunk base prompt: a single point at the bottom centre. In a skyward
        # capture this is reliably on the trunk.
        prompt_points = np.array([[w // 2, int(h * 0.95)]], dtype=np.float32)
        prompt_labels = np.array([1], dtype=np.int32)
    elif prompt_labels is None:
        prompt_labels = np.ones(len(prompt_points), dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=prompt_points.astype(np.float32),
        point_labels=prompt_labels.astype(np.int32),
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)


# ---------------------------------------------------------------------------
# Shared mask utilities
# ---------------------------------------------------------------------------

def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Persist a boolean mask as a single-channel PNG (0/255)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def load_mask(path: str | Path) -> np.ndarray:
    """Load a mask written by `save_mask` and return as bool."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return img > 127


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Boolean-mask IoU. Returns 1.0 if both masks are empty."""
    a = a.astype(bool); b = b.astype(bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b)) / float(union)


# ---------------------------------------------------------------------------
# Bark-color mask in LAB chrominance space with Mahalanobis gating and a
# vertical spatial prior — designed for skyward orbital captures where the
# trunk reliably sits near the frame's horizontal midpoint.
#
# This is an upgrade over a raw HSV-hue threshold for three reasons:
#   1. LAB's a*/b* are chrominance only — luminance (sun vs shadow) doesn't
#      shift them the way it shifts HSV's V channel, so the same bark
#      passes the gate under varied lighting around the orbit.
#   2. Mahalanobis distance defines an oriented ellipse over the sampled
#      bark distribution, capturing the natural correlation between a* and
#      b* (bark is reddish-yellow, so samples cluster along a diagonal
#      rather than an axis-aligned box).
#   3. The vertical spatial prior suppresses bark-colored pixels at the
#      frame periphery (background trees, brown buildings) — a free lever
#      for this specific capture geometry.
# ---------------------------------------------------------------------------

def bark_color_mask_lab(
    image_bgr: np.ndarray,
    sample_grid: tuple = (
        (0.3, 0.85), (0.5, 0.85), (0.7, 0.85),
        (0.3, 0.60), (0.5, 0.60), (0.7, 0.60),
        (0.4, 0.35), (0.5, 0.35), (0.6, 0.35),
        (0.5, 0.15),
    ),
    sample_half: int = 25,
    mahalanobis_max: float = 5.0,
    spatial_sigma_frac: float = 0.35,
    feather_px: int = 15,
    bark_a_min: int = 135,
    bark_b_min: int = 135,
    min_valid_samples: int = 2,
    morph_kernel: int = 5,
    dilate_kernel: int = 13,
    cov_regularization: float = 25.0,
    use_texture: bool = True,
    texture_window: int = 15,
    texture_strength_cap: float = 40.0,
    texture_weight: float = 0.7,
) -> np.ndarray:
    """Per-frame bark mask in LAB chrominance with a vertical spatial prior.

    Args:
        image_bgr: (H, W, 3) BGR frame.
        sample_grid: (x_frac, y_frac) sample-patch centres, in [0, 1]. The
            default 10 positions cover plausible trunk locations along the
            full vertical extent of the frame, biased to the centre column.
        sample_half: half-side of each sample patch in pixels (50x50 default).
        mahalanobis_max: max Mahalanobis distance (in sigmas, computed in
            LAB-ab space) for a pixel to count as bark. 5.0 keeps a wide
            ellipse around the sampled bark distribution — bark varies a
            lot between sunlit, shadowed, and branch-junction regions, so a
            tight 3-sigma window discards most of the trunk. Raise to 7-8
            if the mask is still too narrow, lower to 4 if it bleeds.
        spatial_sigma_frac: width of the vertical Gaussian band centred at
            x = w/2, as a fraction of image width. 0.35 (sigma = ~0.35*w)
            keeps ~95% of the central two-thirds horizontally and only
            cuts the extreme periphery (background trees / buildings).
        feather_px: Gaussian-blur kernel for the final mask edge. Reduced
            to 15px so the binary threshold passes through more of the
            actual mask interior.
        bark_a_min, bark_b_min: median (a*, b*) gate on whether a grid
            sample counts as bark (128 = neutral, > 128 = red for a*,
            > 128 = yellow for b*). 125 is permissive enough to catch
            shadowed bark where chrominance approaches neutral.
        morph_kernel: kernel size for the open/close speckle filter.
        dilate_kernel: kernel size for the final dilation that bridges
            gaps and expands the mask outwards over thin bark areas
            (default 13: larger than the 9 used previously, which left
            holes around branching transitions).
        cov_regularization: ridge added to the LAB-ab covariance matrix
            before inversion. 25.0 corresponds to ~5-pixel a*/b* slack,
            which is enough to make the Mahalanobis ellipse cover the
            full bark distribution even when sampled patches happen to
            be tightly clustered (e.g. all sunlit, no shadow). Raise to
            soften the mask further, lower to tighten.

    Returns:
        (H, W) float mask in [0, 1]. ~1.0 = bark, ~0.0 = not bark or far
        from the central column. Threshold to bool with `mask > 0.5` for
        passing to COLMAP or as a SIFT mask.
    """
    h, w = image_bgr.shape[:2]
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    ab = img_lab[:, :, 1:3].astype(np.float32)   # (H, W, 2)

    # Collect (a*, b*) samples from grid points that look bark-like.
    bark_pixels = []
    for sx, sy in sample_grid:
        cx, cy = int(w * sx), int(h * sy)
        s = sample_half
        patch = img_lab[max(0, cy-s):cy+s, max(0, cx-s):cx+s, 1:3]
        if patch.size == 0:
            continue
        a_med = float(np.median(patch[:, :, 0]))
        b_med = float(np.median(patch[:, :, 1]))
        # Reject samples that aren't plausibly bark (sky, grass, concrete).
        if a_med < bark_a_min or b_med < bark_b_min:
            continue
        bark_pixels.append(patch.reshape(-1, 2).astype(np.float32))

    if len(bark_pixels) < min_valid_samples:
        # Too few grid points landed on confidently-bark-colored pixels.
        # Common causes: sun-glare, lens flare, overexposure, or the
        # trunk drifted out of the sample grid. Returning an empty or
        # tightly-fitted mask here is dangerous — if only 1 sample
        # passed and it happened to land on a non-bark pixel, the
        # Mahalanobis distribution centres on the wrong colour and the
        # mask inverts (bark gets rejected, sky/glare passes). Safer
        # fallback: return an all-pass mask so COLMAP just runs SIFT on
        # the full frame for this image.
        mask_float = np.ones((h, w), dtype=np.float32)
        return mask_float
    else:
        samples = np.vstack(bark_pixels)
        mean = samples.mean(axis=0)
        # Wider covariance: each sample patch is fairly homogeneous, so the
        # raw covariance underestimates bark variance across the whole trunk.
        # Adding a ridge of cov_regularization expands the Mahalanobis
        # ellipse to cover bark colours we didn't happen to sample.
        cov = np.cov(samples.T) + np.eye(2) * cov_regularization
        cov_inv = np.linalg.inv(cov)
        diff = ab - mean[None, None, :]                          # (H, W, 2)
        m2 = np.einsum("hwi,ij,hwj->hw", diff, cov_inv, diff)    # Mahalanobis^2
        binary = (m2 < mahalanobis_max ** 2).astype(np.uint8) * 255

    # Morphological cleanup.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.dilate(
        binary,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel, dilate_kernel)),
    )

    mask_float = binary.astype(np.float32) / 255.0

    # Texture-rescue path: bark has a distinctive vertical-grooved texture
    # that survives lighting variation (sunlit and shadowed bark differ in
    # *colour* but both have the same gradient density at fine scales).
    # The colour-based mask above is brittle to lighting; texture is not.
    # We compute a local gradient-density map and OR it into the mask so
    # that bark pixels rejected by the colour test get rescued if they
    # have bark-like texture. Sky and sun-glare regions have near-zero
    # gradient density and are unaffected.
    if use_texture:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        # Local gradient density: average magnitude in a `texture_window`-sized
        # box around each pixel. Bark patches have many small gradients;
        # sky has almost none.
        local_texture = cv2.boxFilter(grad_mag, cv2.CV_32F,
                                        (texture_window, texture_window))
        # Normalise into [0, 1] with a soft cap. Anything above
        # texture_strength_cap is treated as full texture.
        texture_score = np.clip(local_texture / texture_strength_cap, 0.0, 1.0)
        # OR-combine: pixels passing colour stay at 1.0; pixels rejected
        # by colour get rescued up to texture_weight if their texture is high.
        mask_float = np.maximum(mask_float, texture_score * texture_weight)

    # Vertical spatial prior — Gaussian band centred on x = w/2. Applied
    # AFTER texture rescue so background-tree bark/foliage at the frame
    # periphery still gets suppressed even though it has bark-like texture.
    if spatial_sigma_frac > 0:
        sigma_x = max(1.0, w * spatial_sigma_frac)
        xs = np.arange(w, dtype=np.float32)
        prior = np.exp(-((xs - w / 2.0) ** 2) / (2.0 * sigma_x ** 2))   # (W,)
        mask_float = mask_float * prior[None, :]

    if feather_px > 0:
        # Kernel size must be odd.
        k_feather = feather_px if feather_px % 2 == 1 else feather_px + 1
        mask_float = cv2.GaussianBlur(mask_float, (k_feather, k_feather), 0)

    return mask_float


def save_binary_mask_for_colmap(
    mask_float: np.ndarray,
    out_path: str | Path,
    threshold: float = 0.5,
) -> None:
    """Save a float mask in [0, 1] as a binary 0/255 PNG for COLMAP.

    COLMAP's `--ImageReader.mask_path` expects PNG masks with the same
    basename as the corresponding image (extension swapped to .png). White
    = use this pixel for feature extraction, black = ignore.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    binary = (mask_float >= threshold).astype(np.uint8) * 255
    cv2.imwrite(str(out_path), binary)
