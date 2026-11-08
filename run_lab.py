#!/usr/bin/env python3
"""
CV Lab 2 — Stereo depth (POCO X6 Pro 5G handheld pair)
======================================================
Matches the calibrated phone setup from your capture notes:

  Device   : POCO X6 Pro 5G main (wide) camera
  35mm eq. : 25 mm
  Baseline : B = 12 cm  (6 cm left of centre + 6 cm right of centre)
  Focal f  : f_pixel = (25 * image_width) / 36

Pipeline (exactly as specified):
  1. Load left / right stereo images
  2. SIFT feature detection + description
  3. Brute-Force matching (L2)
  4. RANSAC Homography — align / rectify right onto left rows
  5. StereoSGBM — dense disparity d (horizontal pixel shift)
  6. Triangulation  Z = (f * B) / d   (centimetres)
  7. Export disparity map, depth map, bottle distance

    python run_lab.py
    python run_lab.py --show
    python run_lab.py --max-edge 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Capture / phone calibration (from POCO X6 Pro 5G conversation)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
LEFT_IMAGE = ROOT / "left.jpg"
RIGHT_IMAGE = ROOT / "right.jpg"
OUTPUT_DIR = ROOT / "outputs"

# Total sideways shift: 6 cm left + 6 cm right from centre
BASELINE_CM = 12.0

# POCO X6 Pro main camera 35mm-equivalent focal length
FOCAL_35MM_EQ = 25.0
FILM_WIDTH_MM = 36.0


def pixel_focal_length(image_width: int) -> float:
    """f_pixel = (25 * W) / 36  — POCO main camera, matches chat formula."""
    return (FOCAL_35MM_EQ * float(image_width)) / FILM_WIDTH_MM


def load_pair(left_path: Path, right_path: Path, max_long_edge: int | None):
    img_l = cv2.imread(str(left_path))
    img_r = cv2.imread(str(right_path))
    if img_l is None or img_r is None:
        raise FileNotFoundError(f"Could not read {left_path} / {right_path}")
    if img_l.shape != img_r.shape:
        raise ValueError(f"Shape mismatch: {img_l.shape} vs {img_r.shape}")

    if max_long_edge is not None:
        h, w = img_l.shape[:2]
        if max(h, w) > max_long_edge:
            scale = max_long_edge / float(max(h, w))
            size = (int(round(w * scale)), int(round(h * scale)))
            img_l = cv2.resize(img_l, size, interpolation=cv2.INTER_AREA)
            img_r = cv2.resize(img_r, size, interpolation=cv2.INTER_AREA)

    return img_l, img_r


def sift_ransac_homography_align(img_l, img_r, max_matches: int = 100):
    """
    Handheld alignment (from chat):
      SIFT -> BFMatcher (L2, crossCheck) -> RANSAC Homography
    Warps the right image onto the left image plane so rows line up.
    """
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    h, w = gray_l.shape

    print("2. SIFT feature detection & description")
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)
    if des_l is None or des_r is None:
        raise RuntimeError("SIFT failed to find descriptors")
    print(f"   keypoints: left={len(kp_l)}  right={len(kp_r)}")

    print("3. Brute-Force matching (L2) + keep best matches")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_l, des_r)
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    print(f"   using top {len(matches)} matches")

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    print("4. RANSAC Homography (align right -> left plane)")
    H, mask = cv2.findHomography(pts_r, pts_l, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        raise RuntimeError("RANSAC Homography failed")
    inliers = int(mask.ravel().sum())
    print(f"   RANSAC inliers: {inliers}/{len(matches)}")

    img_r_rect = cv2.warpPerspective(img_r, H, (w, h))
    gray_r_rect = cv2.cvtColor(img_r_rect, cv2.COLOR_BGR2GRAY)

    # Match visualisation (inliers green)
    inlier_matches = [m for m, keep in zip(matches, mask.ravel()) if keep]
    match_vis = cv2.drawMatches(
        img_l, kp_l, img_r, kp_r, inlier_matches[:80], None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return {
        "img_r_rect": img_r_rect,
        "gray_l": gray_l,
        "gray_r_rect": gray_r_rect,
        "H": H,
        "n_inliers": inliers,
        "n_matches": len(matches),
        "match_vis": match_vis,
    }


def compute_sgbm(gray_l, gray_r):
    """StereoSGBM exactly as in the chat script (phone-tuned)."""
    print("5. StereoSGBM disparity (horizontal pixel displacement)")
    block = 7
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,   # divisible by 16; supports closer objects
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    return disparity


def disparity_to_depth_cm(disparity, f_pixel: float, baseline_cm: float):
    """Z = (f * B) / d  with Z, B in centimetres."""
    print("6. Stereo triangulation  Z = (f * B) / d")
    valid = disparity > 0
    depth_cm = np.zeros_like(disparity, dtype=np.float32)
    depth_cm[valid] = (f_pixel * baseline_cm) / disparity[valid]
    return depth_cm


def bottle_distance_cm(depth_cm, h: int, w: int):
    """Median depth in centre ROI (bottle region from chat)."""
    roi_ymin, roi_ymax = int(h * 0.25), int(h * 0.75)
    roi_xmin, roi_xmax = int(w * 0.40), int(w * 0.60)
    roi = depth_cm[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    valid = roi[(roi > 10) & (roi < 500)]
    dist = float(np.median(valid)) if valid.size else float("nan")
    return dist, (roi_xmin, roi_ymin, roi_xmax, roi_ymax)


def colourise_depth(depth_cm, max_view_cm: float = 400.0):
    visual = depth_cm.copy()
    visual[visual > max_view_cm] = 0
    norm = cv2.normalize(visual, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def colourise_disparity(disparity):
    valid = disparity.copy()
    valid[valid < 0] = 0
    norm = cv2.normalize(valid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm, cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def save_fig(path: Path, bgr, title: str, show: bool):
    fig, ax = plt.subplots(figsize=(9, 6))
    if bgr.ndim == 2:
        ax.imshow(bgr, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    for attempt in range(3):
        try:
            fig.savefig(str(path), dpi=140, bbox_inches="tight")
            break
        except OSError:
            path = path.with_name(f"{path.stem}_v{attempt + 2}{path.suffix}")
    else:
        plt.close(fig)
        raise
    print(f"   saved {path.name}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def draw_scanlines(left_bgr, right_bgr, n: int = 12):
    h, w = left_bgr.shape[:2]
    canvas = np.hstack([left_bgr, right_bgr])
    for i in range(n):
        y = int((i + 1) * h / (n + 1))
        cv2.line(canvas, (0, y), (2 * w - 1, y), (0, 255, 0), 1)
    return canvas


def run(left_path, right_path, output_dir, max_long_edge, baseline_cm, show):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== POCO X6 Pro Stereo Depth (SIFT + RANSAC H + SGBM) ===\n")

    print("1. LOAD left / right stereo images")
    img_l, img_r = load_pair(Path(left_path), Path(right_path), max_long_edge)
    h, w = img_l.shape[:2]
    f_pixel = pixel_focal_length(w)
    print(f"   left={Path(left_path).name}  right={Path(right_path).name}")
    print(f"   resolution: {w}x{h}")
    print(f"--- Calibration Specs ---")
    print(f"   POCO X6 Pro 35mm-eq focal : {FOCAL_35MM_EQ} mm")
    print(f"   f_pixel = (25 * {w}) / 36  = {f_pixel:.2f} px")
    print(f"   baseline B                 = {baseline_cm:.1f} cm "
          f"(6 cm left + 6 cm right from centre)")

    align = sift_ransac_homography_align(img_l, img_r)
    img_r_rect = align["img_r_rect"]
    gray_l = align["gray_l"]
    gray_r_rect = align["gray_r_rect"]

    disparity = compute_sgbm(gray_l, gray_r_rect)
    valid_pct = 100.0 * float(np.count_nonzero(disparity > 0)) / disparity.size
    med_d = float(np.median(disparity[disparity > 0])) if np.any(disparity > 0) else float("nan")
    print(f"   disparity valid={valid_pct:.1f}%  median d={med_d:.2f} px")

    depth_cm = disparity_to_depth_cm(disparity, f_pixel, baseline_cm)
    estimated_cm, roi = bottle_distance_cm(depth_cm, h, w)
    rx0, ry0, rx1, ry1 = roi

    print("7. Bottle / target depth measurement")
    if np.isfinite(estimated_cm):
        print("=======================================================")
        print(f"MEASURED DISTANCE TO TARGET: {estimated_cm:.2f} cm")
        print("=======================================================")
    else:
        print("   Warning: could not isolate bottle depth in centre ROI")

    # --- exports (npy + images), as in the chat script ---
    print("8. SAVE arrays + visualisations")
    np.save(str(output_dir / "disparity_map.npy"), disparity)
    np.save(str(output_dir / "depth_map_cm.npy"), depth_cm)
    print("   saved disparity_map.npy")
    print("   saved depth_map_cm.npy")

    disp_gray, disp_color = colourise_disparity(disparity)
    depth_color = colourise_depth(depth_cm)

    cv2.imwrite(str(output_dir / "output_rectified_right.png"), img_r_rect)
    cv2.imwrite(str(output_dir / "output_disparity.png"), disp_gray)
    cv2.imwrite(str(output_dir / "output_disparity_color.png"), disp_color)
    cv2.imwrite(str(output_dir / "output_depth_map.png"), depth_color)
    print("   saved output_rectified_right.png")
    print("   saved output_disparity.png / output_disparity_color.png")
    print("   saved output_depth_map.png")

    # labelled left with ROI + distance
    overlay = img_l.copy()
    cv2.rectangle(overlay, (rx0, ry0), (rx1, ry1), (0, 255, 0), 3)
    if np.isfinite(estimated_cm):
        cv2.putText(
            overlay, f"Dist: {estimated_cm:.1f} cm",
            (rx0, max(30, ry0 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )

    save_fig(output_dir / "01_stereo_pair.png", np.hstack([img_l, img_r]),
             "Input: Left / Right (POCO X6 Pro, B=12 cm)", show)
    save_fig(output_dir / "01b_sift_ransac_matches.png", align["match_vis"],
             "SIFT + BF + RANSAC Homography inliers", show)
    save_fig(output_dir / "02_rectified_scanlines.png",
             draw_scanlines(img_l, img_r_rect),
             "Left + Homography-aligned right (scan lines)", show)
    save_fig(output_dir / "03_target_roi.png", overlay,
             "Left image — bottle ROI + measured distance", show)
    save_fig(output_dir / "04_disparity_gray.png", disp_gray,
             "Disparity map (brighter = larger d = closer)", show)
    save_fig(output_dir / "05_disparity_color.png", disp_color,
             "Disparity map (colour)", show)
    save_fig(output_dir / "06_depth.png", depth_color,
             f"Depth Z = (f * B) / d   f={f_pixel:.0f}px  B={baseline_cm}cm",
             show)

    lines = [
        "CV Lab 2 — POCO X6 Pro 5G stereo depth",
        "",
        "=== Device / calibration (from capture notes) ===",
        "Phone                 : POCO X6 Pro 5G main (wide) camera",
        f"35mm-equivalent f    : {FOCAL_35MM_EQ} mm",
        f"Baseline B           : {baseline_cm} cm  (6 cm left + 6 cm right of centre)",
        f"Image resolution     : {w} x {h}",
        f"f_pixel = (25*W)/36  : {f_pixel:.4f} px",
        "",
        "=== Algorithms ===",
        "1. SIFT               — scale-invariant keypoints/descriptors",
        "2. BFMatcher (L2)     — feature correspondence",
        "3. RANSAC Homography  — handheld tilt correction / row alignment",
        "4. StereoSGBM         — dense disparity d (pixel shift)",
        "5. Triangulation      — Z = (f * B) / d  (cm)",
        "",
        f"RANSAC H inliers     : {align['n_inliers']}/{align['n_matches']}",
        f"Disparity valid %    : {valid_pct:.2f}",
        f"Median disparity (px): {med_d}",
        f"Bottle distance (cm) : {estimated_distance_str(estimated_cm)}",
        "",
        "=== Outputs ===",
        "disparity_map.npy     — raw disparity (px)",
        "depth_map_cm.npy      — depth per pixel (cm)",
        "output_disparity.png / output_disparity_color.png",
        "output_depth_map.png / output_rectified_right.png",
        "01..06 visualisation PNGs",
    ]
    (output_dir / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("   saved results.txt")
    print(f"\nDone. Outputs -> {output_dir}\n")


def estimated_distance_str(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.2f}"


def main():
    p = argparse.ArgumentParser(
        description="POCO X6 Pro stereo depth: SIFT + RANSAC H + SGBM + Z=fB/d"
    )
    p.add_argument("--left", type=Path, default=LEFT_IMAGE)
    p.add_argument("--right", type=Path, default=RIGHT_IMAGE)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument("--baseline-cm", type=float, default=BASELINE_CM,
                   help="Total left-right shift in cm (default 12 = 6+6)")
    p.add_argument("--max-edge", type=int, default=None,
                   help="Optional downscale long edge (default: full resolution)")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    run(args.left, args.right, args.output, args.max_edge,
        args.baseline_cm, args.show)


if __name__ == "__main__":
    main()
