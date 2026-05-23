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
    raise NotImplementedError("classical_sky_mask: from-scratch implementation pending.")
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
