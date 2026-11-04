#!/usr/bin/env python3
"""
CV Lab 2 - Stereo Vision and Depth Estimation
=============================================
Core only: load stereo pair -> rectify -> disparity (BM + SGBM) -> depth.

    python run_lab.py
    python run_lab.py --show
    python run_lab.py --full-res

Equation:  Z = (f * B) / d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
LEFT_IMAGE = ROOT / "left.jpg"
RIGHT_IMAGE = ROOT / "right.jpg"
OUTPUT_DIR = ROOT / "outputs"

MAX_LONG_EDGE = 800

# Experimental placeholders (not ground-truth calibration)
FOCAL_PX = 700.0
BASELINE_M = 0.08

NUM_DISPARITIES = 32  # must be multiple of 16
BM_BLOCK = 15
SGBM_BLOCK = 7


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_pair(left_path: Path, right_path: Path, max_long_edge: int | None):
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None or right is None:
        raise FileNotFoundError("Could not read left.jpg / right.jpg")
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")

    scale = 1.0
    if max_long_edge is not None:
        h, w = left.shape[:2]
        if max(h, w) > max_long_edge:
            scale = max_long_edge / float(max(h, w))
            size = (int(round(w * scale)), int(round(h * scale)))
            left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)

    print("1. LOAD")
    print(f"   left={left_path.name}  right={right_path.name}  shape={left.shape}")
    return left, right, scale


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def match_offsets(left_g, right_g):
    """ORB matches -> median dx, dy (left - right)."""
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(left_g, None)
    k2, d2 = orb.detectAndCompute(right_g, None)
    if d1 is None or d2 is None:
        return 0.0, 0.0
    matches = sorted(
        cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2),
        key=lambda m: m.distance,
    )[:200]
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    dx = pts1[:, 0] - pts2[:, 0]
    dy = pts1[:, 1] - pts2[:, 1]
    return float(np.median(dx)), float(np.median(dy))


def rectify(left_bgr, right_bgr):
    """
    Align rows with a vertical shift from feature matches.
    If disparity trend is negative, swap left/right so d > 0.
    """
    print("2. RECTIFY")
    dx, dy = match_offsets(to_gray(left_bgr), to_gray(right_bgr))
    print(f"   raw median dx={dx:.1f}px  dy={dy:.1f}px")

    if dx < 0:
        print("   swapping left/right for positive disparity")
        left_bgr, right_bgr = right_bgr.copy(), left_bgr.copy()
        dx, dy = -dx, -dy

    h, w = left_bgr.shape[:2]
    shift = int(round(dy))
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    right_rect = cv2.warpAffine(
        right_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    left_rect = left_bgr.copy()
    print(f"   vertical shift on right image: {shift}px")
    return left_rect, right_rect, dx


def fixed_to_float(disp_fixed):
    disp = disp_fixed.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def compute_bm(left_g, right_g, num_disp):
    block = BM_BLOCK if BM_BLOCK % 2 == 1 else BM_BLOCK + 1
    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
    stereo.setUniquenessRatio(15)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)
    return fixed_to_float(stereo.compute(left_g, right_g))


def compute_sgbm(left_g, right_g, num_disp):
    block = SGBM_BLOCK if SGBM_BLOCK % 2 == 1 else SGBM_BLOCK + 1
    mode = getattr(cv2, "STEREO_SGBM_MODE_HH", 0)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        uniquenessRatio=12,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,
        preFilterCap=63,
        mode=mode,
    )
    return fixed_to_float(stereo.compute(left_g, right_g))


def disparity_to_depth(disp, f=FOCAL_PX, B=BASELINE_M):
    """Z = (f * B) / d  — invalid d -> NaN."""
    depth = np.full(disp.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > 0)
    depth[valid] = (f * B) / disp[valid]
    depth[(depth < 0.05) | (depth > 50.0)] = np.nan
    return depth


def colourise(values):
    valid = np.isfinite(values)
    out = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    vals = values[valid]
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((values - lo) / (hi - lo), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0)
    coloured = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    coloured[~valid] = 0
    return coloured


def draw_guides(left_bgr, right_bgr, n=10):
    canvas = np.hstack([left_bgr, right_bgr]).copy()
    h = left_bgr.shape[0]
    for y in np.linspace(0, h - 1, n, dtype=int):
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (0, 255, 0), 1)
    return canvas


def save_fig(path, bgr_or_rgb, title, show, is_bgr=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    img = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB) if is_bgr else bgr_or_rgb
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(path), dpi=140, bbox_inches="tight")
    print(f"   saved {path.name}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run(left_path, right_path, output_dir, max_long_edge, show):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Stereo Vision + Depth ===\n")
    left, right, _ = load_pair(Path(left_path), Path(right_path), max_long_edge)
    left_r, right_r, dx_med = rectify(left, right)

    left_g, right_g = to_gray(left_r), to_gray(right_r)
    num_disp = max(16, int(np.ceil((abs(dx_med) + 16) / 16.0) * 16))
    num_disp = min(num_disp, 64)
    # fallback if estimate is tiny
    if num_disp < NUM_DISPARITIES:
        num_disp = NUM_DISPARITIES

    print("3. DISPARITY")
    print(f"   numDisparities={num_disp}")
    disp_bm = compute_bm(left_g, right_g, num_disp)
    disp_sgbm = compute_sgbm(left_g, right_g, num_disp)

    def stats(name, d):
        v = d[np.isfinite(d)]
        if v.size == 0:
            print(f"   {name}: no valid pixels")
            return
        print(
            f"   {name}: valid={v.size}/{d.size} "
            f"median={np.median(v):.2f}px  "
            f"range=[{v.min():.2f}, {v.max():.2f}]"
        )

    stats("StereoBM", disp_bm)
    stats("StereoSGBM", disp_sgbm)

    print("4. DEPTH  Z = (f * B) / d")
    print(f"   f={FOCAL_PX:.1f}px  B={BASELINE_M:.3f}m  (experimental)")
    depth = disparity_to_depth(disp_sgbm)
    dv = depth[np.isfinite(depth)]
    if dv.size:
        print(
            f"   valid depth pixels={dv.size}  "
            f"median Z={np.median(dv):.3f}m ({np.median(dv)*100:.1f}cm)"
        )
        print("   NOTE: depths are illustrative without real calibration.")
    else:
        print("   no valid depth pixels")

    print("5. SAVE")
    # original
    save_fig(
        output_dir / "01_stereo_pair.png",
        np.hstack([left, right]),
        "Original stereo pair",
        show,
    )
    # rectified + guides
    save_fig(
        output_dir / "02_rectified.png",
        draw_guides(left_r, right_r),
        "Rectified pair (horizontal guides)",
        show,
    )
    # disparity
    save_fig(
        output_dir / "03_disparity_bm.png",
        colourise(disp_bm),
        "StereoBM disparity",
        show,
    )
    save_fig(
        output_dir / "04_disparity_sgbm.png",
        colourise(disp_sgbm),
        "StereoSGBM disparity",
        show,
    )
    # depth
    save_fig(
        output_dir / "05_depth.png",
        colourise(depth),
        "Depth map from SGBM  Z=(f*B)/d",
        show,
    )
    cv2.imwrite(str(output_dir / "05_depth_raw.png"), colourise(depth))

    # text summary
    summary = output_dir / "results.txt"
    med_d = float(np.nanmedian(disp_sgbm)) if np.isfinite(disp_sgbm).any() else None
    med_z = float(np.median(dv)) if dv.size else None
    lines = [
        "CV Lab 2 - Stereo Vision & Depth",
        f"f (px)              : {FOCAL_PX}",
        f"B (m)               : {BASELINE_M}",
        f"numDisparities      : {num_disp}",
        f"SGBM median d (px)  : {med_d}",
        f"Median depth (m)    : {med_z}",
        f"Median depth (cm)   : {None if med_z is None else med_z * 100}",
        "Equation             : Z = (f * B) / d",
        "NOTE: experimental f,B — depths are illustrative only.",
    ]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"   saved {summary.name}")
    print(f"\nDone. Outputs in: {output_dir}\n")


def main():
    p = argparse.ArgumentParser(description="Stereo vision + depth (core)")
    p.add_argument("--left", type=Path, default=LEFT_IMAGE)
    p.add_argument("--right", type=Path, default=RIGHT_IMAGE)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument("--show", action="store_true")
    p.add_argument("--full-res", action="store_true")
    args = p.parse_args()

    run(
        args.left,
        args.right,
        args.output,
        None if args.full_res else MAX_LONG_EDGE,
        args.show,
    )


if __name__ == "__main__":
    main()
