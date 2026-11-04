"""
Visualisation and result export helpers.

All figures are saved under ``outputs/`` and optionally shown interactively
when ``show=True``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .bottle_detection import BottleDetection, BottleDepthStats, draw_bottle_overlay
from .depth import DepthResult, metres_to_cm
from .disparity import DisparityResult, colourise_disparity
from .image_io import StereoPair
from .rectification import RectificationResult, draw_horizontal_guides


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_fig(fig: plt.Figure, path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_cv_image(path: Path, bgr: np.ndarray) -> None:
    cv2.imwrite(str(path), bgr)
    print(f"  Saved: {path.name}")


def visualise_original(pair: StereoPair, out_dir: Path, show: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].imshow(_bgr_to_rgb(pair.left_bgr))
    axes[0].set_title("Left image")
    axes[0].axis("off")
    axes[1].imshow(_bgr_to_rgb(pair.right_bgr))
    axes[1].set_title("Right image")
    axes[1].axis("off")
    fig.suptitle("Original Stereo Pair")
    _save_fig(fig, out_dir / "01_original_stereo.png", show)
    save_cv_image(out_dir / "01_left.png", pair.left_bgr)
    save_cv_image(out_dir / "01_right.png", pair.right_bgr)


def visualise_rectified(
    rect: RectificationResult, out_dir: Path, show: bool = False
) -> None:
    guides = draw_horizontal_guides(rect.left_rect_bgr, rect.right_rect_bgr)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(_bgr_to_rgb(guides))
    ax.set_title(f"Rectified Stereo Pair with Epipolar Guides\n({rect.method})")
    ax.axis("off")
    _save_fig(fig, out_dir / "02_rectified_stereo.png", show)
    save_cv_image(out_dir / "02_rectified_guides.png", guides)
    save_cv_image(out_dir / "02_left_rectified.png", rect.left_rect_bgr)
    save_cv_image(out_dir / "02_right_rectified.png", rect.right_rect_bgr)


def visualise_disparity(
    bm: DisparityResult,
    sgbm: DisparityResult,
    out_dir: Path,
    show: bool = False,
) -> None:
    bm_color = colourise_disparity(bm.disparity_px)
    sgbm_color = colourise_disparity(sgbm.disparity_px)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(_bgr_to_rgb(bm_color))
    axes[0].set_title("StereoBM disparity")
    axes[0].axis("off")
    axes[1].imshow(_bgr_to_rgb(sgbm_color))
    axes[1].set_title("StereoSGBM disparity")
    axes[1].axis("off")
    fig.suptitle("Disparity Map Comparison")
    _save_fig(fig, out_dir / "03_disparity_comparison.png", show)

    # Greyscale normalised maps
    def _grey(d: np.ndarray) -> np.ndarray:
        valid = np.isfinite(d)
        g = np.zeros(d.shape, dtype=np.uint8)
        if np.any(valid):
            vals = d[valid]
            lo, hi = np.percentile(vals, [2, 98])
            scaled = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            scaled = np.nan_to_num(scaled, nan=0.0)
            g = (scaled * 255).astype(np.uint8)
        return g

    save_cv_image(out_dir / "03_disparity_bm_gray.png", _grey(bm.disparity_px))
    save_cv_image(out_dir / "03_disparity_sgbm_gray.png", _grey(sgbm.disparity_px))
    save_cv_image(out_dir / "04_disparity_bm_color.png", bm_color)
    save_cv_image(out_dir / "04_disparity_sgbm_color.png", sgbm_color)

    fig2, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(_bgr_to_rgb(sgbm_color))
    ax.set_title("Colour-mapped StereoSGBM Disparity")
    ax.axis("off")
    _save_fig(fig2, out_dir / "04_colour_mapped_disparity.png", show)


def visualise_bottle(
    image_bgr: np.ndarray,
    det: BottleDetection,
    out_dir: Path,
    show: bool = False,
) -> np.ndarray:
    overlay = draw_bottle_overlay(image_bgr, det)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(_bgr_to_rgb(overlay))
    axes[0].set_title("Bottle detection + bounding box")
    axes[0].axis("off")
    axes[1].imshow(det.mask, cmap="gray")
    axes[1].set_title("Segmentation mask")
    axes[1].axis("off")
    fig.suptitle("Blue Bottle Detection")
    _save_fig(fig, out_dir / "05_bottle_detection.png", show)
    save_cv_image(out_dir / "05_bottle_overlay.png", overlay)
    save_cv_image(out_dir / "05_bottle_mask.png", det.mask)
    return overlay


def visualise_depth(
    depth: DepthResult,
    det: BottleDetection,
    stats: BottleDepthStats,
    overlay_bgr: np.ndarray,
    out_dir: Path,
    show: bool = False,
) -> None:
    depth_m = depth.depth_m
    valid = np.isfinite(depth_m)

    # Colour depth map
    colour = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if np.any(valid):
        vals = depth_m[valid]
        lo, hi = np.percentile(vals, [5, 95])
        scaled = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0, 1)
        scaled = np.nan_to_num(scaled, nan=0.0)
        colour = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        colour[~valid] = 0

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(_bgr_to_rgb(colour))
    ax.set_title(
        f"Depth map from {depth.source_disparity}\n"
        f"{'(EXPERIMENTAL - not metric GT)' if not depth.is_metric else '(metric)'}"
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="relative depth")
    _save_fig(fig, out_dir / "06_depth_map.png", show)
    save_cv_image(out_dir / "06_depth_map_color.png", colour)

    # Bottle-specific depth visualisation
    bottle_depth_vis = overlay_bgr.copy()
    if det.found and det.mask is not None:
        tint = colour.copy()
        # Show coloured depth only inside the bottle, keep RGB elsewhere.
        m = det.mask > 0
        bottle_depth_vis = overlay_bgr.copy()
        bottle_depth_vis[m] = cv2.addWeighted(overlay_bgr, 0.35, tint, 0.65, 0)[m]

        label_y = 30
        if stats.median_depth_m is not None:
            txt = (
                f"Median Z ~ {stats.median_depth_m:.2f} m "
                f"({metres_to_cm(stats.median_depth_m):.1f} cm)"
            )
            cv2.putText(
                bottle_depth_vis, txt, (10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
            label_y += 24
        if stats.centre_depth_m is not None:
            txt = (
                f"Centre Z ~ {stats.centre_depth_m:.2f} m "
                f"({metres_to_cm(stats.centre_depth_m):.1f} cm)"
            )
            cv2.putText(
                bottle_depth_vis, txt, (10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
        if not depth.is_metric:
            cv2.putText(
                bottle_depth_vis, "EXPERIMENTAL DEPTH", (10, label_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )

    fig2, ax2 = plt.subplots(figsize=(6, 7))
    ax2.imshow(_bgr_to_rgb(bottle_depth_vis))
    ax2.set_title("Bottle-specific Depth Visualisation")
    ax2.axis("off")
    _save_fig(fig2, out_dir / "07_bottle_depth.png", show)
    save_cv_image(out_dir / "07_bottle_depth_overlay.png", bottle_depth_vis)


def write_results_txt(
    path: Path,
    pair: StereoPair,
    calib,
    det: BottleDetection,
    stats: BottleDepthStats,
    depth: DepthResult,
) -> None:
    """Write the numerical summary required for the lab report."""
    lines = []
    lines.append("Computer Vision Lab 2 - Stereo Vision & Depth Estimation")
    lines.append("=" * 64)
    lines.append(f"Left image          : {pair.left_path.name}")
    lines.append(f"Right image         : {pair.right_path.name}")
    lines.append(f"Working resolution  : {pair.width} x {pair.height} "
                 f"(scale={pair.scale:.4f})")
    lines.append("")
    lines.append("Calibration")
    lines.append("-" * 64)
    lines.append(f"Mode                : {calib.mode}")
    lines.append(f"Focal length f (px) : {depth.focal_px:.4f}")
    lines.append(f"Baseline B (m)      : {depth.baseline_m:.6f}")
    lines.append(f"Metric depths?      : {depth.is_metric}")
    for note in calib.notes:
        lines.append(f"NOTE: {note}")
    lines.append("")
    lines.append("Bottle detection")
    lines.append("-" * 64)
    lines.append(f"Detected            : {det.found}")
    if det.found and det.bbox is not None and det.centroid is not None:
        x, y, w, h = det.bbox
        cx, cy = det.centroid
        lines.append(f"Bounding box (x,y,w,h): ({x}, {y}, {w}, {h})")
        lines.append(f"Centroid (x, y)       : ({cx}, {cy})")
        lines.append(f"Contour area (px)     : {det.area_px}")
    lines.append("")
    lines.append("Bottle depth")
    lines.append("-" * 64)
    lines.append(f"Disparity source      : {depth.source_disparity}")
    lines.append(f"Valid bottle pixels   : {stats.valid_pixels}")
    lines.append(f"Median disparity (px) : {stats.median_disparity_px}")
    lines.append(f"Centre disparity (px) : {stats.centre_disparity_px}")
    if stats.median_depth_m is not None:
        lines.append(
            f"Median depth           : {stats.median_depth_m:.4f} m "
            f"({metres_to_cm(stats.median_depth_m):.2f} cm)"
        )
    else:
        lines.append("Median depth           : unavailable")
    if stats.centre_depth_m is not None:
        lines.append(
            f"Centre depth           : {stats.centre_depth_m:.4f} m "
            f"({metres_to_cm(stats.centre_depth_m):.2f} cm)"
        )
    else:
        lines.append("Centre depth           : unavailable")
    lines.append("")
    lines.append("Equation used:  Z = (f * B) / d")
    if not depth.is_metric:
        lines.append(
            "WARNING: Experimental calibration - do NOT treat these depths "
            "as ground-truth metric measurements."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {path.name}")


def print_final_summary(
    det: BottleDetection,
    stats: BottleDepthStats,
    depth: DepthResult,
    calib,
) -> None:
    print("=" * 60)
    print("9. RESULTS SUMMARY")
    print("=" * 60)
    if det.found and det.bbox is not None and det.centroid is not None:
        x, y, w, h = det.bbox
        cx, cy = det.centroid
        print(f"  Bottle bbox (x,y,w,h) : ({x}, {y}, {w}, {h})")
        print(f"  Bottle centroid       : ({cx}, {cy})")
    else:
        print("  Bottle: NOT DETECTED")
    print(f"  Median disparity (px) : {stats.median_disparity_px}")
    print(f"  Centre disparity (px) : {stats.centre_disparity_px}")
    print(f"  Baseline B (m)        : {depth.baseline_m:.4f}")
    print(f"  Focal length f (px)   : {depth.focal_px:.2f}")
    if stats.median_depth_m is not None:
        print(f"  Estimated median depth: {stats.median_depth_m:.3f} m "
              f"({metres_to_cm(stats.median_depth_m):.1f} cm)")
    if stats.centre_depth_m is not None:
        print(f"  Estimated centre depth: {stats.centre_depth_m:.3f} m "
              f"({metres_to_cm(stats.centre_depth_m):.1f} cm)")
    print(f"  Calibration mode      : {calib.mode}")
    if not depth.is_metric:
        print("  *** Depths are EXPERIMENTAL / ILLUSTRATIVE only ***")