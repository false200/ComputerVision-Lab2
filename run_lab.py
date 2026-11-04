#!/usr/bin/env python3
"""
CV Lab 2 - Simple Stereo | Camera Calibration concepts
======================================================
Covers the lecture pipeline:

  1) Perspective projection (left/right cameras)
  2) Depth & disparity:  z = (b * fx) / (u_l - u_r)
  3) Correspondence / stereo matching
  4) Window-based template matching along the scan line
  5) Similarity metrics: SAD, SSD, NCC
  6) Large + adaptive windows
  7) Disparity map (closer = brighter) and depth map

    python run_lab.py
    python run_lab.py --show
    python run_lab.py --full-res

---------------------------------------------------------------------------
LECTURE FORMULAS
---------------------------------------------------------------------------
Perspective projection:
  (u_l, v_l) = (fx * x/z + ox,  fy * y/z + oy)
  (u_r, v_r) = (fx * (x-b)/z + ox,  fy * y/z + oy)

Solving for 3-D (Simple Stereo: Depth and Disparity):
  d = u_l - u_r                         # disparity
  z = (b * fx) / d                      # depth  (highlighted in lecture)
  x = b * (u_l - ox) / d
  y = b * fx * (v_l - oy) / (fy * d)

Window-based matching (along same scan line after rectification):
  Slide template window T on left along search scan line L on right.
  Pick (k,l) that minimises SAD/SSD or maximises NCC.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Config  (fx, fy, b, ox, oy — lecture "known" parameters; approximate here)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
LEFT_IMAGE = ROOT / "left.jpg"
RIGHT_IMAGE = ROOT / "right.jpg"
OUTPUT_DIR = ROOT / "outputs"

MAX_LONG_EDGE = 800

FX = FY = 700.0          # fx, fy  (pixels)
BASELINE_B = 0.08        # b      (metres)
# ox, oy set to image centre after load/rectify

NUM_DISPARITIES = 48     # search range along scan line (multiple of 16 for BM)

# Window sizes (lecture: small=noisy, large=smooth but blurry edges)
WINDOW_SMALL = 5
WINDOW_LARGE = 31        # ~30 px as in lecture slide
WINDOW_ADAPTIVE = (5, 15, 31)  # adaptive: try several, keep best

# Keep aliases used below
FOCAL_PX = FX
BASELINE_M = BASELINE_B


# =============================================================================
# Similarity metrics (lecture: SAD, SSD, NCC)
# =============================================================================
def sad(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of Absolute Differences — minimise."""
    return float(np.sum(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def ssd(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of Squared Differences — minimise."""
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sum(d * d))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Cross-Correlation — maximise."""
    a32 = a.astype(np.float32).ravel()
    b32 = b.astype(np.float32).ravel()
    denom = float(np.sqrt(np.sum(a32 * a32) * np.sum(b32 * b32)))
    if denom < 1e-6:
        return -1.0
    return float(np.sum(a32 * b32) / denom)


# =============================================================================
# Load / rectify helpers
# =============================================================================
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


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def orb_offsets(left_g, right_g):
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(left_g, None)
    k2, d2 = orb.detectAndCompute(right_g, None)
    matches = sorted(
        cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2),
        key=lambda m: m.distance,
    )[:250]
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    dx = float(np.median(pts1[:, 0] - pts2[:, 0]))
    dy = float(np.median(pts1[:, 1] - pts2[:, 1]))
    return dx, dy


def rectify(left_bgr, right_bgr):
    """
    Rectify so corresponding points share the same row (scan line).
    Then window search is only horizontal: template T slides on line L.
    """
    print("2. RECTIFY (make epipolar lines = horizontal scan lines)")
    dx, dy = orb_offsets(to_gray(left_bgr), to_gray(right_bgr))
    print(f"   feature dx={dx:.1f}px  dy={dy:.1f}px")
    if dx < 0:
        print("   swap left/right so disparity d = u_l - u_r > 0")
        left_bgr, right_bgr = right_bgr.copy(), left_bgr.copy()
        dx, dy = -dx, -dy

    h, w = left_bgr.shape[:2]
    shift = int(round(dy))
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    right_r = cv2.warpAffine(
        right_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    print(f"   vertical align dy={shift}px  (same-line matching ready)")
    return left_bgr.copy(), right_r, dx


# =============================================================================
# Window-based matching along the scan line
# =============================================================================
def match_along_scanline(
    left_g: np.ndarray,
    right_g: np.ndarray,
    u: int,
    v: int,
    win: int,
    num_disp: int,
    metric: str = "sad",
):
    """
    Lecture demo: take template T around (u,v) in left image E_l,
    slide it along search scan line L in right image E_r (same row v),
    return best disparity and scores for SAD/SSD/NCC.
    """
    r = win // 2
    h, w = left_g.shape
    if u - r < 0 or u + r >= w or v - r < 0 or v + r >= h:
        return None

    T = left_g[v - r:v + r + 1, u - r:u + r + 1]
    best_d, best_cost = 0, None
    # also compute all three metrics at the winning SAD location for the report
    for d in range(0, min(num_disp, u - r)):
        ur = u - d
        patch = right_g[v - r:v + r + 1, ur - r:ur + r + 1]
        if patch.shape != T.shape:
            continue
        if metric == "sad":
            cost = sad(T, patch)
            better = best_cost is None or cost < best_cost
        elif metric == "ssd":
            cost = ssd(T, patch)
            better = best_cost is None or cost < best_cost
        else:  # ncc
            cost = ncc(T, patch)
            better = best_cost is None or cost > best_cost
        if better:
            best_cost, best_d = cost, d

    ur = u - best_d
    patch = right_g[v - r:v + r + 1, ur - r:ur + r + 1]
    return {
        "d": best_d,
        "sad": sad(T, patch),
        "ssd": ssd(T, patch),
        "ncc": ncc(T, patch),
        "u": u,
        "v": v,
        "win": win,
    }


def compute_bm(left_g, right_g, num_disp, block_size: int):
    """
    OpenCV StereoBM = dense window-based matching using SAD along each scan line.
    block_size is the template window T (must be odd >= 5).
    """
    block = block_size if block_size % 2 == 1 else block_size + 1
    block = int(np.clip(block, 5, 255))
    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
    stereo.setPreFilterCap(31)
    stereo.setTextureThreshold(5)
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)
    stereo.setDisp12MaxDiff(2)
    disp = stereo.compute(left_g, right_g).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def compute_sgbm(left_g, right_g, num_disp, block_size: int):
    block = block_size if block_size % 2 == 1 else block_size + 1
    block = int(np.clip(block, 3, 21))
    mode = getattr(cv2, "STEREO_SGBM_MODE_HH", 0)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        uniquenessRatio=8,
        speckleWindowSize=150,
        speckleRange=32,
        disp12MaxDiff=2,
        preFilterCap=63,
        mode=mode,
    )
    disp = stereo.compute(left_g, right_g).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def adaptive_window_disparity(left_g, right_g, num_disp, windows=WINDOW_ADAPTIVE):
    """
    Lecture adaptive window method:
      For each point, match with several window sizes and keep a robust result.
    We run SAD-based StereoBM at multiple sizes and take the per-pixel median
    (stable when sizes disagree; reduces noise of small windows and blur of large).
    """
    print(f"   Adaptive windows T = {list(windows)} (SAD along scan lines)")
    maps = [compute_bm(left_g, right_g, num_disp, w) for w in windows]
    stack = np.stack(maps, axis=0)
    return np.nanmedian(stack, axis=0)


def filter_disparity(disp: np.ndarray) -> np.ndarray:
    out = disp.copy()
    fixed = np.zeros(disp.shape, dtype=np.int16)
    valid = np.isfinite(disp) & (disp > 0)
    fixed[valid] = (disp[valid] * 16.0).astype(np.int16)
    cv2.filterSpeckles(fixed, newVal=0, maxSpeckleSize=200, maxDiff=32)
    cleaned = fixed.astype(np.float32) / 16.0
    cleaned[cleaned <= 0] = np.nan
    fill = cleaned.copy()
    med = np.nanmedian(fill) if np.isfinite(fill).any() else 0.0
    fill[~np.isfinite(fill)] = med
    blur = cv2.medianBlur(fill.astype(np.float32), 5)
    mask = np.isfinite(cleaned)
    cleaned[mask] = blur[mask]
    return cleaned


# =============================================================================
# Depth / 3-D from lecture formulas
# =============================================================================
def disparity_to_xyz(disp, fx=FX, fy=FY, b=BASELINE_B):
    """
    Solve for (x, y, z) from perspective stereo:

      z = (b * fx) / d
      x = b * (u_l - ox) / d
      y = b * fx * (v_l - oy) / (fy * d)
    """
    h, w = disp.shape
    ox, oy = w / 2.0, h / 2.0
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    z = np.full(disp.shape, np.nan, dtype=np.float32)
    x = np.full(disp.shape, np.nan, dtype=np.float32)
    y = np.full(disp.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(disp) & (disp > 0)
    d = disp[valid]
    z[valid] = (b * fx) / d
    x[valid] = b * (uu[valid] - ox) / d
    y[valid] = b * fx * (vv[valid] - oy) / (fy * d)

    # reject absurd depths
    bad = (z < 0.05) | (z > 50.0)
    z[bad] = np.nan
    x[bad] = np.nan
    y[bad] = np.nan
    return x, y, z


# =============================================================================
# Visualisation
# =============================================================================
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


def disparity_gray_map(disp: np.ndarray) -> np.ndarray:
    """Disparity Map: closer=brighter, farther=darker (lecture GT style)."""
    gray = np.zeros(disp.shape, dtype=np.uint8)
    valid = np.isfinite(disp) & (disp > 0)
    if np.any(valid):
        vals = disp[valid]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((disp - lo) / (hi - lo), 0, 1)
        gray = (np.nan_to_num(scaled, nan=0.0) * 255).astype(np.uint8)
        gray[~valid] = 0
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_guides(left_bgr, right_bgr, n=10):
    canvas = np.hstack([left_bgr, right_bgr]).copy()
    h = left_bgr.shape[0]
    for y in np.linspace(0, h - 1, n, dtype=int):
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (0, 255, 0), 1)
    return canvas


def draw_window_scanline_demo(left_bgr, right_bgr, u, v, win, d):
    """Visualise template T on left and search scan line L on right (lecture slide)."""
    demo = np.hstack([left_bgr.copy(), right_bgr.copy()])
    h, w = left_bgr.shape[:2]
    r = win // 2
    # template T (red) on left
    cv2.rectangle(demo, (u - r, v - r), (u + r, v + r), (0, 0, 255), 2)
    # scan line L (white) on right
    cv2.rectangle(demo, (w, v - r), (2 * w - 1, v + r), (255, 255, 255), 2)
    # matched window on right
    ur = u - d
    cv2.rectangle(demo, (w + ur - r, v - r), (w + ur + r, v + r), (0, 255, 0), 2)
    cv2.line(demo, (0, v), (2 * w - 1, v), (0, 255, 255), 1)
    cv2.putText(demo, "T (left)", (u - r, max(20, v - r - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(demo, "L scan line (right)", (w + 10, max(20, v - r - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(demo, f"match d={d}px", (w + ur, min(h - 10, v + r + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return demo


def save_fig(path, bgr, title, show):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(path), dpi=140, bbox_inches="tight")
    print(f"   saved {path.name}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def stats(name, d):
    v = d[np.isfinite(d)]
    if v.size == 0:
        print(f"   {name}: no valid pixels")
        return None
    med = float(np.median(v))
    print(f"   {name}: valid={100.0 * v.size / d.size:.1f}%  median d={med:.2f}px")
    return med


# =============================================================================
# Main pipeline
# =============================================================================
def run(left_path, right_path, output_dir, max_long_edge, show):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Simple Stereo (lecture checklist) ===\n")
    left, right, _ = load_pair(Path(left_path), Path(right_path), max_long_edge)
    left_r, right_r, dx_med = rectify(left, right)

    left_g = enhance_gray(to_gray(left_r))
    right_g = enhance_gray(to_gray(right_r))
    h, w = left_g.shape
    ox, oy = w / 2.0, h / 2.0

    num_disp = max(16, int(np.ceil((abs(dx_med) + 24) / 16.0) * 16))
    num_disp = int(np.clip(num_disp, NUM_DISPARITIES, 96))

    # --- Window-based methods ---
    print("3. WINDOW-BASED MATCHING (template T along scan line L)")
    print(f"   numDisparities (search on L) = {num_disp}")
    print(f"   known params: fx=fy={FX}, b={BASELINE_B}, ox={ox:.1f}, oy={oy:.1f}")

    print(f"   Small window T={WINDOW_SMALL} (sensitive to noise)")
    disp_small = filter_disparity(compute_bm(left_g, right_g, num_disp, WINDOW_SMALL))
    print(f"   Large window T={WINDOW_LARGE} (smoother, poorer localization)")
    disp_large = filter_disparity(compute_bm(left_g, right_g, num_disp, WINDOW_LARGE))
    print("   Adaptive window method (multiple sizes, median fusion)")
    disp_adapt = filter_disparity(
        adaptive_window_disparity(left_g, right_g, num_disp, WINDOW_ADAPTIVE)
    )
    disp_sgbm = filter_disparity(compute_sgbm(left_g, right_g, num_disp, 11))

    stats("BM small", disp_small)
    stats("BM large", disp_large)
    med_adapt = stats("BM adaptive", disp_adapt)
    med_sgbm = stats("SGBM", disp_sgbm)

    # Single-point lecture demo: SAD/SSD/NCC on one template
    u0, v0 = w // 2, h // 2
    demo = match_along_scanline(
        left_g, right_g, u0, v0, win=WINDOW_LARGE, num_disp=num_disp, metric="sad"
    )
    if demo:
        print("   Scan-line template demo at image centre:")
        print(f"      SAD={demo['sad']:.1f}  SSD={demo['ssd']:.1f}  "
              f"NCC={demo['ncc']:.3f}  -> d={demo['d']}px")

    # Primary disparity for depth = adaptive SAD windows
    disp = disp_adapt

    print("4. DEPTH & 3-D  (z = b*fx / d)")
    x_map, y_map, z_map = disparity_to_xyz(disp, FX, FY, BASELINE_B)
    zv = z_map[np.isfinite(z_map)]
    med_z = float(np.median(zv)) if zv.size else None
    if med_z is not None:
        print(f"   median z={med_z:.3f}m ({med_z * 100:.1f}cm)")
        print("   also computed x,y maps from lecture formulas")
    print("   NOTE: fx,b approximate (no chessboard) — illustrative metres")

    print("5. SAVE")
    save_fig(output_dir / "01_stereo_pair.png", np.hstack([left, right]),
             "Left / Right camera images", show)
    save_fig(output_dir / "02_rectified_scanlines.png", draw_guides(left_r, right_r),
             "Rectified pair — matching along horizontal scan lines", show)

    if demo:
        save_fig(
            output_dir / "02b_window_scanline_demo.png",
            draw_window_scanline_demo(left_r, right_r, demo["u"], demo["v"],
                                      demo["win"], demo["d"]),
            "Window-based method: template T (red) + search line L (white)",
            show,
        )

    # small vs large window (lecture comparison)
    save_fig(
        output_dir / "03_window_small_vs_large.png",
        np.hstack([disparity_gray_map(disp_small), disparity_gray_map(disp_large)]),
        f"Window size={WINDOW_SMALL} (noise) | Window size={WINDOW_LARGE} (smooth)",
        show,
    )

    # adaptive + SGBM
    gray = disparity_gray_map(disp)
    save_fig(output_dir / "04_disparity_adaptive_gray.png", gray,
             "Disparity Map — adaptive windows (closer=brighter, farther=darker)", show)
    save_fig(output_dir / "04b_disparity_sgbm_gray.png", disparity_gray_map(disp_sgbm),
             "Disparity Map — StereoSGBM (closer=brighter)", show)
    save_fig(output_dir / "05_disparity_color.png", colourise(disp),
             "Disparity (colour) from adaptive SAD windows", show)
    save_fig(output_dir / "06_depth.png", colourise(z_map),
             "Depth z = (b * fx) / d", show)

    cv2.imwrite(str(output_dir / "04_disparity_gray_raw.png"),
                cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY))

    lines = [
        "CV Lab 2 - Simple Stereo checklist (lecture)",
        "",
        "=== Perspective projection ===",
        "(u_l, v_l) = (fx*x/z + ox, fy*y/z + oy)",
        "(u_r, v_r) = (fx*(x-b)/z + ox, fy*y/z + oy)",
        "",
        "=== Solve for (x,y,z) ===",
        "d = u_l - u_r",
        "z = (b * fx) / d",
        "x = b * (u_l - ox) / d",
        "y = b * fx * (v_l - oy) / (fy * d)",
        "",
        "=== Window-based matching ===",
        "Template T on left slides along scan line L on right",
        "Metrics: SAD (min), SSD (min), NCC (max)",
        "StereoBM implements dense SAD along scan lines",
        f"Small window = {WINDOW_SMALL} | Large window = {WINDOW_LARGE}",
        f"Adaptive windows = {list(WINDOW_ADAPTIVE)}",
        "",
        f"fx=fy (px)              : {FX}",
        f"b (m)                   : {BASELINE_B}",
        f"ox, oy                  : {ox:.2f}, {oy:.2f}",
        f"numDisparities          : {num_disp}",
        f"Adaptive median d (px)  : {med_adapt}",
        f"SGBM median d (px)      : {med_sgbm}",
        f"Median depth z (m)      : {med_z}",
        f"Median depth z (cm)     : {None if med_z is None else med_z * 100}",
    ]
    if demo:
        lines += [
            "",
            "Centre template demo:",
            f"  SAD={demo['sad']:.2f}  SSD={demo['ssd']:.2f}  NCC={demo['ncc']:.4f}  d={demo['d']}",
        ]
    lines += [
        "",
        "Gray disparity: brighter = closer, darker = farther",
        "NOTE: without chessboard calibration, metres are approximate.",
        "Tsukuba-style GT needs a calibrated stereo rig + controlled capture.",
    ]
    (output_dir / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("   saved results.txt")
    print(f"\nDone. Outputs -> {output_dir}\n")


def main():
    p = argparse.ArgumentParser(description="Simple Stereo lab (lecture checklist)")
    p.add_argument("--left", type=Path, default=LEFT_IMAGE)
    p.add_argument("--right", type=Path, default=RIGHT_IMAGE)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument("--show", action="store_true")
    p.add_argument("--full-res", action="store_true")
    args = p.parse_args()
    run(args.left, args.right, args.output,
        None if args.full_res else MAX_LONG_EDGE, args.show)


if __name__ == "__main__":
    main()
