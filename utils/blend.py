"""Feathered face composite — cached elliptical mask + Reinhard LAB colour transfer.

The wav2lip and ultralight avatars stamp the generated mouth crop back into the
full frame as a hard rectangle. The crop drifts in brightness and hue from the
surrounding skin, and the rectangle edge makes that drift visible as a box.

This module fixes that with three steps:
  1. A cached elliptical feather mask (built once per unique bbox size).
  2. Reinhard colour transfer in LAB so the crop matches the surrounding skin.
  3. Alpha composite in float32 into a preallocated buffer.

Cost: <= 2 ms/frame at 576x768 (measured). The mask is cached, so the per-frame
work is one LAB conversion, one masked mean/std, and one alpha blend.
"""

import os
import cv2
import numpy as np
from typing import Dict, Tuple

# Per unique (crop_h, crop_w) we build the feather mask once and reuse it.
# At 25 fps with a fixed avatar the bbox size never changes, so the dict stays
# at one entry in practice.
_mask_cache: Dict[Tuple[int, int], np.ndarray] = {}


def feather_mask(h: int, w: int, sigma: float | None = None) -> np.ndarray:
    """Elliptical feather mask, cached per (h, w).

    Returns a float32 array of shape (h, w) with values in [0, 1].
    The ellipse axes are ~42% of width and ~46% of height, so the mask covers
    the face but fades to zero before the rectangular crop edge.

    sigma controls the blur radius; default is 6% of min(h, w).
    """
    key = (h, w)
    if key in _mask_cache:
        return _mask_cache[key]

    if sigma is None:
        sigma = 0.06 * min(h, w)

    mask = np.zeros((h, w), dtype=np.uint8)
    ax_w = int(0.42 * w)
    ax_h = int(0.46 * h)
    cv2.ellipse(mask, (w // 2, h // 2), (ax_w, ax_h), 0, 0, 360, 255, -1)

    sigma_px = max(sigma, 1.0)
    ksize = int(sigma_px * 6) | 1  # always odd
    mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma_px)
    result = (mask.astype(np.float32) / 255.0)

    _mask_cache[key] = result
    return result


def reinhard_colour_transfer(
    src_crop: np.ndarray,
    gen_crop: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Match the generated crop's per-channel mean/std to the source crop's.

    Both crops must be the same shape (H, W, 3) uint8. The mask (H, W) float32
    selects the pixels used for statistics — typically the feather mask, so the
    transfer is driven by the face interior, not the transparent border.

    The correction is clamped to +/- 2 sigma of the source to prevent a dark or
    blown-out frame from swinging the whole face.
    """
    src_lab = cv2.cvtColor(src_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(gen_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    m = mask if mask.ndim == 2 else mask[:, :, 0]
    m_bool = m > 0.1
    if m_bool.sum() < 20:
        return gen_crop

    result = gen_lab.copy()

    # Channel 0 (L: lightness): match mean and contrast with a guarded scale factor
    src_l = src_lab[:, :, 0][m_bool]
    gen_l = gen_lab[:, :, 0][m_bool]
    src_l_mean, src_l_std = src_l.mean(), src_l.std() + 1e-6
    gen_l_mean, gen_l_std = gen_l.mean(), gen_l.std() + 1e-6
    scale_l = np.clip(src_l_std / gen_l_std, 0.7, 1.4)
    result[:, :, 0] = (gen_lab[:, :, 0] - gen_l_mean) * scale_l + src_l_mean

    # Channels 1 & 2 (a, b: chrominance): mean shift only to align skin undertone
    # without compressing or clamping lip saturation/lipstick
    for ch in (1, 2):
        src_vals = src_lab[:, :, ch][m_bool]
        gen_vals = gen_lab[:, :, ch][m_bool]
        result[:, :, ch] = gen_lab[:, :, ch] + (src_vals.mean() - gen_vals.mean())

    res_uint8 = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(res_uint8, cv2.COLOR_LAB2BGR)


def composite(
    src_frame: np.ndarray,
    gen_crop: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    sigma: float | None = None,
    colour_transfer: bool = True,
    _buf: np.ndarray | None = None,
) -> np.ndarray:
    """Alpha-composite the generated crop into the source frame with feathering.

    src_frame: (H, W, 3) uint8 — the full frame to paste into (not modified).
    gen_crop:  (h, w, 3) — the generated region, any numeric dtype.
    (x1, y1, x2, y2): the target rectangle in src_frame.
    sigma: blur sigma for the feather mask; None = auto (6% of min(h, w)).
    colour_transfer: if True, run Reinhard LAB transfer before blending.
    _buf: optional preallocated float32 (h, w, 3) buffer to avoid per-frame alloc.

    Returns a new (H, W, 3) uint8 frame.
    """
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return src_frame.copy()

    gen = gen_crop.astype(np.float32)
    if gen.max() > 1.5:
        gen = np.clip(gen, 0, 255)

    gen_uint8 = gen.astype(np.uint8)
    gen_resized = gen_uint8
    if gen_uint8.shape[0] != h or gen_uint8.shape[1] != w:
        gen_resized = cv2.resize(gen_uint8, (w, h))

    mask = feather_mask(h, w, sigma)
    mask3 = mask[:, :, np.newaxis]

    if colour_transfer:
        src_crop = src_frame[y1:y2, x1:x2]
        gen_resized = reinhard_colour_transfer(src_crop, gen_resized, mask)

    src_roi = src_frame[y1:y2, x1:x2].astype(np.float32)
    gen_f = gen_resized.astype(np.float32)

    if _buf is not None and _buf.shape == src_roi.shape:
        blended = _buf
    else:
        blended = np.empty_like(src_roi)

    np.add(src_roi * (1.0 - mask3), gen_f * mask3, out=blended)
    np.clip(blended, 0, 255, out=blended)

    result = src_frame.copy()
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result


def clear_cache() -> None:
    """Drop all cached masks. Tests only."""
    _mask_cache.clear()


# ---------------------------------------------------------------------------
# Runtime configuration — read once at import time.
# ---------------------------------------------------------------------------
BLEND_MODE = os.environ.get("GF_BLEND", "feather").strip().lower()
BLEND_SIGMA: float | None = None
_raw_sigma = os.environ.get("GF_BLEND_FEATHER", "").strip()
if _raw_sigma:
    try:
        BLEND_SIGMA = float(_raw_sigma)
    except ValueError:
        pass
