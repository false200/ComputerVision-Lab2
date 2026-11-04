"""
Blue water-bottle detection via HSV colour segmentation.

Pipeline
--------
1. Convert BGR → HSV.
2. Build a binary mask from one or more blue/cyan hue ranges.
3. Morphological open/close to remove speckles and fill holes.
4. Find contours; keep the largest blob that exceeds a minimum area.
5. Return bounding box, centroid and the cleaned mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .depth import DepthResult, metres_to_cm
from .disparity import DisparityResult


@dataclass
class BottleDetection:
    found: bool
    mask: np.ndarray
    contour: np.ndarray | None
    bbox: tuple[int, int, int, int] | None  # x, y, w, h
    centroid: tuple[int, int] | None
    area_px: int = 0


@dataclass
class BottleDepthStats:
    median_disparity_px: float | None
    centre_disparity_px: float | None
    median_depth_m: float | None
    centre_depth_m: float | None
    valid_pixels: int


def detect_blue_bottle(image_bgr: np.ndarray) -> BottleDetection:
    """Detect the dominant blue bottle region in a BGR image."""
    print("=" * 60)
    print("6. BOTTLE DETECTION (HSV + morphology + contours)")
    print("=" * 60)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in config.HSV_BLUE_RANGES:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))

    k = config.MORPH_KERNEL_SIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = config.MIN_BOTTLE_AREA_RATIO * mask.shape[0] * mask.shape[1]

    # Prefer tall, bottle-like blobs (height > width) to reject floor reflections
    # or wide false-positive regions after aggressive warps.
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / float(max(w, 1))
        # Score: area rewarded, extreme aspect mildly preferred, huge width penalised.
        width_penalty = 1.0 if w < 0.55 * mask.shape[1] else 0.15
        score = area * min(aspect, 4.0) * width_penalty
        candidates.append((score, area, cnt, (x, y, w, h)))

    if not candidates:
        print("  Bottle NOT detected - adjust HSV ranges if needed.")
        return BottleDetection(
            found=False,
            mask=mask,
            contour=None,
            bbox=None,
            centroid=None,
            area_px=0,
        )

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, best_area, best, (x, y, w, h) = candidates[0]
    M = cv2.moments(best)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    # Refine mask to the chosen contour only (ignore other blue blobs).
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [best], -1, 255, thickness=-1)

    print(f"  Bottle found: bbox=(x={x}, y={y}, w={w}, h={h}), "
          f"centroid=({cx}, {cy}), area={int(best_area)} px")
    return BottleDetection(
        found=True,
        mask=clean,
        contour=best,
        bbox=(x, y, w, h),
        centroid=(cx, cy),
        area_px=int(best_area),
    )


def draw_bottle_overlay(image_bgr: np.ndarray, det: BottleDetection) -> np.ndarray:
    """Draw bounding box, contour and centroid on a copy of the image."""
    out = image_bgr.copy()
    if not det.found or det.bbox is None:
        cv2.putText(
            out, "Bottle not detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
        )
        return out

    x, y, w, h = det.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if det.contour is not None:
        cv2.drawContours(out, [det.contour], -1, (0, 255, 255), 2)
    if det.centroid is not None:
        cx, cy = det.centroid
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            out, "Bottle", (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
    return out


def bottle_depth_stats(
    det: BottleDetection,
    disparity: DisparityResult,
    depth: DepthResult,
) -> BottleDepthStats:
    """
    Compute median / centre disparity and depth inside the bottle mask.

    Background and invalid pixels are ignored via the detection mask and
    NaN filtering.
    """
    print("=" * 60)
    print("7. BOTTLE DEPTH CALCULATION")
    print("=" * 60)

    empty = BottleDepthStats(None, None, None, None, 0)
    if not det.found or det.centroid is None:
        print("  Skipped - bottle was not detected.")
        return empty

    mask = det.mask > 0
    disp_vals = disparity.disparity_px[mask]
    disp_vals = disp_vals[np.isfinite(disp_vals) & (disp_vals > 0)]

    depth_vals = depth.depth_m[mask]
    depth_vals = depth_vals[np.isfinite(depth_vals)]

    median_disp = float(np.median(disp_vals)) if disp_vals.size else None
    median_depth = float(np.median(depth_vals)) if depth_vals.size else None

    cx, cy = det.centroid
    # Centre disparity: expand a window around the centroid *inside the bottle
    # mask* until enough plausible matches are found. Textureless bottle
    # surfaces often have holes at the geometric centre; tiny noisy
    # disparities (<< scene median) are rejected.
    centre_disp = None
    centre_depth = None
    h, w = disparity.disparity_px.shape
    d_lo = 0.4 * median_disp if median_disp else 2.0
    d_hi = 2.5 * median_disp if median_disp else 1e6
    for radius in (5, 10, 20, 40, 80, 120):
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        local_mask = mask[y0:y1, x0:x1]
        local_disp = disparity.disparity_px[y0:y1, x0:x1]
        local_depth = depth.depth_m[y0:y1, x0:x1]
        plausible = (
            local_mask
            & np.isfinite(local_disp)
            & (local_disp >= d_lo)
            & (local_disp <= d_hi)
        )
        good_d = local_disp[plausible]
        good_z = local_depth[plausible & np.isfinite(local_depth)]
        if good_d.size >= 8:
            centre_disp = float(np.median(good_d))
            if good_z.size:
                centre_depth = float(np.median(good_z))
            elif centre_disp and depth.focal_px > 0 and depth.baseline_m > 0:
                centre_depth = float((depth.focal_px * depth.baseline_m) / centre_disp)
            print(f"  Centre sample radius           : {radius} px "
                  f"({good_d.size} plausible disparity pixels)")
            break

    print(f"  Valid disparity pixels in bottle : {disp_vals.size}")
    print(f"  Median disparity (px)            : {median_disp}")
    print(f"  Centre disparity (px)            : {centre_disp}")
    print(f"  Median depth                     : "
          f"{median_depth} m  ({metres_to_cm(median_depth)} cm)"
          if median_depth is not None else "  Median depth: unavailable")
    print(f"  Centre depth                     : "
          f"{centre_depth} m  ({metres_to_cm(centre_depth)} cm)"
          if centre_depth is not None else "  Centre depth: unavailable")
    if not depth.is_metric:
        print("  NOTE: values inherit EXPERIMENTAL calibration - not ground truth.")

    return BottleDepthStats(
        median_disparity_px=median_disp,
        centre_disparity_px=centre_disp,
        median_depth_m=median_depth,
        centre_depth_m=centre_depth,
        valid_pixels=int(disp_vals.size),
    )