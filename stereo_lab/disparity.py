"""
Disparity computation with StereoBM and StereoSGBM.

OpenCV returns disparity in fixed-point 16-bit format (value / 16.0 = pixels).
We convert to float32 pixel disparity and mark invalid / non-positive entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .rectification import RectificationResult


@dataclass
class DisparityResult:
    name: str
    disparity_px: np.ndarray  # float32, invalid = NaN
    raw_fixed_point: np.ndarray


def _fixed_to_float(disp_fixed: np.ndarray) -> np.ndarray:
    """Convert OpenCV 16-bit fixed-point disparity to float pixels (NaN if ≤ 0)."""
    disp = disp_fixed.astype(np.float32) / 16.0
    disp[disp <= 0.0] = np.nan
    return disp


def compute_stereobm(
    rect: RectificationResult, params: dict | None = None
) -> DisparityResult:
    """Block Matching disparity (local, fast, noisier on textureless regions)."""
    p = {**config.STEREO_BM, **(params or {})}
    # blockSize must be odd and ≥ 5
    block = int(p["blockSize"])
    if block % 2 == 0:
        block += 1
    num_disp = int(p["numDisparities"])
    num_disp = max(16, num_disp - (num_disp % 16))

    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
    stereo.setPreFilterCap(int(p["preFilterCap"]))
    stereo.setTextureThreshold(int(p["textureThreshold"]))
    stereo.setUniquenessRatio(int(p["uniquenessRatio"]))
    stereo.setSpeckleWindowSize(int(p["speckleWindowSize"]))
    stereo.setSpeckleRange(int(p["speckleRange"]))
    stereo.setDisp12MaxDiff(int(p["disp12MaxDiff"]))

    raw = stereo.compute(rect.left_rect_gray, rect.right_rect_gray)
    return DisparityResult(
        name="StereoBM",
        disparity_px=_fixed_to_float(raw),
        raw_fixed_point=raw,
    )


def compute_stereosgbm(
    rect: RectificationResult, params: dict | None = None
) -> DisparityResult:
    """
    Semi-Global Block Matching disparity.

    SGBM aggregates matching costs along multiple paths, producing smoother
    and more accurate maps than plain BM — especially around object edges.
    """
    p = {**config.STEREO_SGBM, **(params or {})}
    block = int(p["blockSize"])
    if block % 2 == 0:
        block += 1
    num_disp = int(p["numDisparities"])
    num_disp = max(16, num_disp - (num_disp % 16))

    # Prefer HH mode when available for better quality on larger images.
    mode = getattr(cv2, "STEREO_SGBM_MODE_HH", 0)

    stereo = cv2.StereoSGBM_create(
        minDisparity=int(p["minDisparity"]),
        numDisparities=num_disp,
        blockSize=block,
        P1=int(p["P1"]),
        P2=int(p["P2"]),
        disp12MaxDiff=int(p["disp12MaxDiff"]),
        uniquenessRatio=int(p["uniquenessRatio"]),
        speckleWindowSize=int(p["speckleWindowSize"]),
        speckleRange=int(p["speckleRange"]),
        preFilterCap=int(p["preFilterCap"]),
        mode=mode,
    )
    raw = stereo.compute(rect.left_rect_gray, rect.right_rect_gray)
    return DisparityResult(
        name="StereoSGBM",
        disparity_px=_fixed_to_float(raw),
        raw_fixed_point=raw,
    )


def compute_both_disparities(
    rect: RectificationResult,
    num_disparities: int | None = None,
) -> tuple[DisparityResult, DisparityResult]:
    """Run StereoBM and StereoSGBM and print a short comparison."""
    print("=" * 60)
    print("4. DISPARITY CALCULATION")
    print("=" * 60)

    overrides = {}
    if num_disparities is not None:
        num_disparities = max(16, int(num_disparities) - (int(num_disparities) % 16))
        overrides["numDisparities"] = num_disparities
        print(f"  Auto-tuned numDisparities = {num_disparities}")

    bm = compute_stereobm(rect, overrides)
    sgbm = compute_stereosgbm(rect, overrides)

    def _stats(d: DisparityResult) -> None:
        valid = d.disparity_px[np.isfinite(d.disparity_px)]
        if valid.size == 0:
            print(f"  {d.name}: NO valid disparities")
            return
        print(
            f"  {d.name}: valid={valid.size}/{d.disparity_px.size} "
            f"({100.0 * valid.size / d.disparity_px.size:.1f}%), "
            f"min={valid.min():.2f}, median={np.median(valid):.2f}, "
            f"max={valid.max():.2f} px"
        )

    _stats(bm)
    _stats(sgbm)
    print("  SGBM is generally preferred for the final depth estimate.")
    return bm, sgbm


def colourise_disparity(disparity_px: np.ndarray) -> np.ndarray:
    """Map disparity to a BGR colour image (invalid → black)."""
    valid = np.isfinite(disparity_px)
    out = np.zeros((*disparity_px.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    vals = disparity_px[valid]
    norm = np.zeros(disparity_px.shape, dtype=np.uint8)
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((disparity_px - lo) / (hi - lo), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0)
    norm = (scaled * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    coloured[~valid] = 0
    return coloured