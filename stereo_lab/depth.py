"""
Depth estimation from disparity.

Core stereo equation (pinhole, rectified cameras):

    Z = (f * B) / d

where
    Z — depth along the optical axis (metres)
    f — focal length in pixels
    B — baseline in metres
    d — disparity in pixels

Invalid, zero and negative disparities are excluded to avoid division by zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import CalibrationResult
from .disparity import DisparityResult


@dataclass
class DepthResult:
    depth_m: np.ndarray  # float32 metres, invalid = NaN
    focal_px: float
    baseline_m: float
    is_metric: bool
    source_disparity: str


def disparity_to_depth(
    disparity: DisparityResult,
    calib: CalibrationResult,
) -> DepthResult:
    """
    Convert a disparity map to a depth map using Z = (f * B) / d.

    NaN is written wherever disparity is missing or non-positive.
    """
    print("=" * 60)
    print("5. DEPTH ESTIMATION  (Z = f * B / d)")
    print("=" * 60)

    f = calib.focal_length_px
    B = calib.baseline_m
    d = disparity.disparity_px

    depth = np.full(d.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    depth[valid] = (f * B) / d[valid]

    # Reject absurd depths (e.g. tiny noisy disparities -> huge Z)
    depth[depth > 50.0] = np.nan
    depth[depth < 0.05] = np.nan

    n_valid = int(np.isfinite(depth).sum())
    print(f"  Using disparity from : {disparity.name}")
    print(f"  Focal length f (px)  : {f:.2f}")
    print(f"  Baseline B (m)       : {B:.4f}")
    print(f"  Valid depth pixels   : {n_valid} / {depth.size}")
    if not calib.is_metric:
        print("  WARNING: experimental calibration - depths are ILLUSTRATIVE only.")
    else:
        print("  Calibration mode provides metric depths (subject to stereo T accuracy).")

    return DepthResult(
        depth_m=depth,
        focal_px=f,
        baseline_m=B,
        is_metric=calib.is_metric,
        source_disparity=disparity.name,
    )


def depth_at_pixel(depth_m: np.ndarray, x: int, y: int, window: int = 5) -> float | None:
    """Median depth in a small window around (x, y). Returns None if no valid pixels."""
    h, w = depth_m.shape
    r = window // 2
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def metres_to_cm(metres: float | None) -> float | None:
    if metres is None:
        return None
    return metres * 100.0