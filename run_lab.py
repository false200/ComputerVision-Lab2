#!/usr/bin/env python3
"""
Computer Vision Lab 2 - Stereo Vision and Depth Estimation
==========================================================
Single-file implementation (OpenCV + NumPy + Matplotlib).

    python run_lab.py
    python run_lab.py --show
    python run_lab.py --full-res
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
LEFT_IMAGE = PROJECT_ROOT / "left.jpg"
RIGHT_IMAGE = PROJECT_ROOT / "right.jpg"
CALIBRATION_DIR = PROJECT_ROOT / "calibration_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MAX_LONG_EDGE = 800
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.025

# Experimental placeholders (NOT ground truth without real calibration)
EXPERIMENTAL_FOCAL_PX = 700.0
EXPERIMENTAL_BASELINE_M = 0.08

STEREO_BM = {
    "numDisparities": 48,
    "blockSize": 15,
    "preFilterCap": 31,
    "textureThreshold": 10,
    "uniquenessRatio": 15,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1,
}
STEREO_SGBM = {
    "minDisparity": 0,
    "numDisparities": 48,
    "blockSize": 7,
    "P1": 8 * 3 * 7 * 7,
    "P2": 32 * 3 * 7 * 7,
    "disp12MaxDiff": 1,
    "uniquenessRatio": 12,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "preFilterCap": 63,
}

HSV_BLUE_RANGES = [
    ((85, 60, 80), (115, 255, 255)),   # cyan / light blue
    ((100, 40, 20), (130, 255, 180)),  # navy
]
MORPH_KERNEL_SIZE = 7
MIN_BOTTLE_AREA_RATIO = 0.005


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class StereoPair:
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    left_path: Path
    right_path: Path
    scale: float = 1.0

    @property
    def height(self) -> int:
        return self.left_bgr.shape[0]

    @property
    def width(self) -> int:
        return self.left_bgr.shape[1]

    @property
    def left_gray(self) -> np.ndarray:
        return cv2.cvtColor(self.left_bgr, cv2.COLOR_BGR2GRAY)

    @property
    def right_gray(self) -> np.ndarray:
        return cv2.cvtColor(self.right_bgr, cv2.COLOR_BGR2GRAY)


@dataclass
class CalibrationResult:
    mode: str
    camera_matrix_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_left: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: tuple[int, int]
    rms_error: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def focal_length_px(self) -> float:
        return float(
            (self.camera_matrix_left[0, 0] + self.camera_matrix_left[1, 1]) / 2.0
        )

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def is_metric(self) -> bool:
        return self.mode == "chessboard"


@dataclass
class RectificationResult:
    left_rect_bgr: np.ndarray
    right_rect_bgr: np.ndarray
    left_rect_gray: np.ndarray
    right_rect_gray: np.ndarray
    method: str
    H1: np.ndarray | None = None
    H2: np.ndarray | None = None
    Q: np.ndarray | None = None
    match_dx_stats: dict | None = None


@dataclass
class DisparityResult:
    name: str
    disparity_px: np.ndarray
    raw_fixed_point: np.ndarray


@dataclass
class DepthResult:
    depth_m: np.ndarray
    focal_px: float
    baseline_m: float
    is_metric: bool
    source_disparity: str


@dataclass
class BottleDetection:
    found: bool
    mask: np.ndarray
    contour: np.ndarray | None
    bbox: tuple[int, int, int, int] | None
    centroid: tuple[int, int] | None
    area_px: int = 0


@dataclass
class BottleDepthStats:
    median_disparity_px: float | None
    centre_disparity_px: float | None
    median_depth_m: float | None
    centre_depth_m: float | None
    valid_pixels: int


# =============================================================================
# 1. LOAD IMAGES
# =============================================================================
def load_stereo_pair(
    left_path: Path | str,
    right_path: Path | str,
    max_long_edge: int | None = None,
) -> StereoPair:
    left_path, right_path = Path(left_path), Path(right_path)
    if not left_path.is_file():
        raise FileNotFoundError(f"Left image not found: {left_path}")
    if not right_path.is_file():
        raise FileNotFoundError(f"Right image not found: {right_path}")

    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(f"OpenCV failed to read left image: {left_path}")
    if right is None:
        raise FileNotFoundError(f"OpenCV failed to read right image: {right_path}")

    print("=" * 60)
    print("1. STEREO IMAGE LOADING")
    print("=" * 60)
    print(f"  Left  : {left_path.name}  shape={left.shape}  dtype={left.dtype}")
    print(f"  Right : {right_path.name}  shape={right.shape}  dtype={right.dtype}")
    if left.shape != right.shape:
        raise ValueError(
            f"Incompatible shapes: left {left.shape} vs right {right.shape}"
        )
    print("  Compatibility check: PASSED (identical dimensions)")

    scale = 1.0
    if max_long_edge is not None:
        h, w = left.shape[:2]
        long_edge = max(h, w)
        if long_edge > max_long_edge:
            scale = max_long_edge / float(long_edge)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            left = cv2.resize(left, new_size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, new_size, interpolation=cv2.INTER_AREA)
            print(f"  Resized for processing: {left.shape}  (scale={scale:.4f})")

    return StereoPair(left, right, left_path, right_path, scale)


# =============================================================================
# 2. CALIBRATION
# =============================================================================
def _list_calibration_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files: list[Path] = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(sorted(folder.glob(pat)))
    return files


def calibrate_from_chessboard(image_size: tuple[int, int]) -> CalibrationResult | None:
    images = _list_calibration_images(CALIBRATION_DIR)
    if len(images) < 5:
        return None

    cols, rows = CHESSBOARD_SIZE
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    obj_points, img_points = [], []
    detected = 0
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != image_size:
            img = cv2.resize(img, image_size)
        found, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE, None)
        if not found:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        detected += 1

    if detected < 5:
        return None

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    R = np.eye(3, dtype=np.float64)
    T = np.array([[EXPERIMENTAL_BASELINE_M], [0.0], [0.0]], dtype=np.float64)
    notes = [
        f"Chessboard intrinsics from {detected} images (RMS={rms:.4f} px).",
        f"Baseline set to experimental value {EXPERIMENTAL_BASELINE_M} m.",
    ]
    return CalibrationResult(
        "chessboard", K.copy(), K.copy(), dist.copy(), dist.copy(),
        R, T, image_size, float(rms), notes,
    )


def build_experimental_calibration(image_size: tuple[int, int]) -> CalibrationResult:
    w, h = image_size
    fx = fy = float(EXPERIMENTAL_FOCAL_PX)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    T = np.array([[EXPERIMENTAL_BASELINE_M], [0.0], [0.0]], dtype=np.float64)
    notes = [
        "EXPERIMENTAL MODE - no chessboard calibration data was found.",
        f"Using placeholder focal length f ~ {fx:.1f} px and baseline "
        f"B ~ {EXPERIMENTAL_BASELINE_M:.3f} m.",
        "Depth values from Z = (f * B) / d are ILLUSTRATIVE ONLY.",
        "For accurate depth you need chessboard intrinsics, stereoCalibrate "
        "extrinsics, and a measured baseline.",
    ]
    return CalibrationResult(
        "experimental", K, K.copy(), dist, dist.copy(), R, T, image_size, None, notes
    )


def obtain_calibration(image_size: tuple[int, int]) -> CalibrationResult:
    print("=" * 60)
    print("2. CAMERA CALIBRATION")
    print("=" * 60)
    result = calibrate_from_chessboard(image_size)
    if result is None:
        print(f"  No usable chessboard images in: {CALIBRATION_DIR}")
        print("  Falling back to EXPERIMENTAL mode.")
        result = build_experimental_calibration(image_size)
    else:
        print("  Chessboard calibration succeeded.")

    print(f"  Mode              : {result.mode}")
    print(f"  Image size (WxH)  : {result.image_size[0]} x {result.image_size[1]}")
    print(f"  Focal length (px) : {result.focal_length_px:.2f}")
    print(f"  Baseline B (m)    : {result.baseline_m:.4f}")
    print("  Left camera matrix K_L:")
    print(result.camera_matrix_left)
    print("  Distortion coeffs (left):")
    print(result.dist_coeffs_left.ravel())
    print("  Rotation R (left->right):")
    print(result.R)
    print("  Translation T (metres):")
    print(result.T.ravel())
    for note in result.notes:
        print(f"  NOTE: {note}")
    return result


# =============================================================================
# 3. RECTIFICATION
# =============================================================================
def _orb_matches(left_g, right_g, keep=250):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(left_g, None)
    kp2, des2 = orb.detectAndCompute(right_g, None)
    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        raise RuntimeError("Not enough ORB features for rectification.")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)
    good = matches[: max(keep, len(matches) // 3)]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, len(matches), len(good)


def _dx_dy_stats(pts1, pts2) -> dict:
    dx = pts1[:, 0] - pts2[:, 0]
    dy = pts1[:, 1] - pts2[:, 1]
    return {
        "dx_median": float(np.median(dx)),
        "dx_p10": float(np.percentile(dx, 10)),
        "dx_p90": float(np.percentile(dx, 90)),
        "dy_median": float(np.median(dy)),
        "dy_std": float(np.std(dy)),
        "n": int(len(dx)),
    }


def _pack(left_bgr, right_bgr, method, **kwargs) -> RectificationResult:
    return RectificationResult(
        left_bgr, right_bgr,
        cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY),
        method, **kwargs,
    )


def _rectify_essential(pair, calib, pts1, pts2) -> RectificationResult | None:
    K, dist = calib.camera_matrix_left, calib.dist_coeffs_left
    w, h = pair.width, pair.height
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None
    inl = mask.ravel().astype(bool)
    if inl.sum() < 20:
        return None
    _, R, T, _ = cv2.recoverPose(E, pts1[inl], pts2[inl], K)
    t_norm = np.linalg.norm(T)
    if t_norm < 1e-8:
        return None
    T = T / t_norm * calib.baseline_m
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist, K, dist, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)
    left_rect = cv2.remap(pair.left_bgr, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(pair.right_bgr, map2x, map2y, cv2.INTER_LINEAR)
    valid_frac = float(np.mean(cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY) > 0))
    if valid_frac < 0.55:
        print(f"  Essential rectification rejected (valid fraction={valid_frac:.2f}).")
        return None
    return _pack(left_rect, right_rect, "essential matrix + stereoRectify", Q=Q)


def _rectify_uncalibrated(pair, pts1, pts2) -> RectificationResult | None:
    h, w = pair.height, pair.width
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None:
        return None
    inl = mask.ravel().astype(bool)
    if inl.sum() < 20:
        return None
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(pts1[inl], pts2[inl], F, (w, h))
    if not ok:
        return None
    left_rect = cv2.warpPerspective(pair.left_bgr, H1, (w, h))
    right_rect = cv2.warpPerspective(pair.right_bgr, H2, (w, h))
    valid_frac = float(np.mean(cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY) > 0))
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H1).reshape(-1, 2)
    drift = float(np.mean(np.linalg.norm(warped - corners.reshape(-1, 2), axis=1)))
    if valid_frac < 0.55 or drift > 0.35 * max(w, h):
        print(
            f"  Uncalibrated rectification rejected "
            f"(valid={valid_frac:.2f}, mean corner drift={drift:.1f}px)."
        )
        return None
    print(f"  Fundamental-matrix inliers: {int(inl.sum())}")
    return _pack(
        left_rect, right_rect,
        "uncalibrated (ORB + F + stereoRectifyUncalibrated)",
        H1=H1, H2=H2,
    )


def _rectify_vertical_align(pair, pts1, pts2) -> RectificationResult:
    stats = _dx_dy_stats(pts1, pts2)
    dy = int(round(stats["dy_median"]))
    print(f"  Applying vertical alignment shift of {dy} px to the right image.")
    print(
        f"  Pre-align disparity stats: dx median={stats['dx_median']:.1f} "
        f"[{stats['dx_p10']:.1f}, {stats['dx_p90']:.1f}], "
        f"dy median={stats['dy_median']:.1f} (std={stats['dy_std']:.1f})"
    )
    h, w = pair.height, pair.width
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    right_rect = cv2.warpAffine(
        pair.right_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    return _pack(
        pair.left_bgr.copy(), right_rect,
        f"vertical translation alignment (dy={dy}px)",
        match_dx_stats=stats,
    )


def _rectify_calibrated(pair, calib) -> RectificationResult:
    w, h = pair.width, pair.height
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        calib.camera_matrix_left, calib.dist_coeffs_left,
        calib.camera_matrix_right, calib.dist_coeffs_right,
        (w, h), calib.R, calib.T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib.camera_matrix_left, calib.dist_coeffs_left, R1, P1, (w, h), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib.camera_matrix_right, calib.dist_coeffs_right, R2, P2, (w, h), cv2.CV_32FC1
    )
    left_rect = cv2.remap(pair.left_bgr, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(pair.right_bgr, map2x, map2y, cv2.INTER_LINEAR)
    return _pack(left_rect, right_rect, "calibrated (stereoRectify + remap)", Q=Q)


def rectify_stereo(pair: StereoPair, calib: CalibrationResult) -> RectificationResult:
    print("=" * 60)
    print("3. STEREO RECTIFICATION")
    print("=" * 60)

    if calib.mode == "chessboard":
        try:
            result = _rectify_calibrated(pair, calib)
            print(f"  Method: {result.method}")
            return result
        except cv2.error as exc:
            print(f"  Calibrated rectification failed ({exc}); trying feature-based.")

    pts1, pts2, n_total, n_kept = _orb_matches(pair.left_gray, pair.right_gray)
    print(f"  Feature matches: {n_total} total, {n_kept} kept for geometry")
    stats = _dx_dy_stats(pts1, pts2)
    print(
        f"  Raw match offsets: dx median={stats['dx_median']:.1f} px, "
        f"dy median={stats['dy_median']:.1f} px"
    )

    # OpenCV stereo matchers expect x_left > x_right for foreground (d > 0).
    # If median dx is negative, treat the files as swapped for this run.
    working = pair
    if stats["dx_median"] < 0:
        print("  Detected negative disparity trend - swapping left/right for matching.")
        working = StereoPair(
            left_bgr=pair.right_bgr.copy(),
            right_bgr=pair.left_bgr.copy(),
            left_path=pair.right_path,
            right_path=pair.left_path,
            scale=pair.scale,
        )
        pts1, pts2, n_total, n_kept = _orb_matches(working.left_gray, working.right_gray)
        stats = _dx_dy_stats(pts1, pts2)
        print(
            f"  After swap: dx median={stats['dx_median']:.1f} px, "
            f"dy median={stats['dy_median']:.1f} px"
        )

    result = _rectify_essential(working, calib, pts1, pts2)
    if result is None:
        result = _rectify_uncalibrated(working, pts1, pts2)
    if result is None:
        result = _rectify_vertical_align(working, pts1, pts2)

    result.match_dx_stats = stats
    print(f"  Method: {result.method}")
    print("  Corresponding points should now lie on roughly the same horizontal rows.")
    return result


def draw_horizontal_guides(left_bgr, right_bgr, num_lines=12) -> np.ndarray:
    h = left_bgr.shape[0]
    canvas = np.hstack([left_bgr, right_bgr]).copy()
    for y in np.linspace(0, h - 1, num_lines, dtype=int):
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (0, 255, 0), 1)
    return canvas


def suggest_num_disparities(rect: RectificationResult, fallback: int = 64) -> int:
    stats = rect.match_dx_stats
    if not stats:
        return fallback
    need = abs(stats["dx_p90"]) + 16
    num = int(np.ceil(need / 16.0) * 16)
    return int(np.clip(num, 16, 160))


# =============================================================================
# 4. DISPARITY (StereoBM + StereoSGBM)
# =============================================================================
def _fixed_to_float(disp_fixed: np.ndarray) -> np.ndarray:
    disp = disp_fixed.astype(np.float32) / 16.0
    disp[disp <= 0.0] = np.nan
    return disp


def compute_stereobm(rect, params=None) -> DisparityResult:
    p = {**STEREO_BM, **(params or {})}
    block = int(p["blockSize"])
    if block % 2 == 0:
        block += 1
    num_disp = max(16, int(p["numDisparities"]) - (int(p["numDisparities"]) % 16))
    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
    stereo.setPreFilterCap(int(p["preFilterCap"]))
    stereo.setTextureThreshold(int(p["textureThreshold"]))
    stereo.setUniquenessRatio(int(p["uniquenessRatio"]))
    stereo.setSpeckleWindowSize(int(p["speckleWindowSize"]))
    stereo.setSpeckleRange(int(p["speckleRange"]))
    stereo.setDisp12MaxDiff(int(p["disp12MaxDiff"]))
    raw = stereo.compute(rect.left_rect_gray, rect.right_rect_gray)
    return DisparityResult("StereoBM", _fixed_to_float(raw), raw)


def compute_stereosgbm(rect, params=None) -> DisparityResult:
    p = {**STEREO_SGBM, **(params or {})}
    block = int(p["blockSize"])
    if block % 2 == 0:
        block += 1
    num_disp = max(16, int(p["numDisparities"]) - (int(p["numDisparities"]) % 16))
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
    return DisparityResult("StereoSGBM", _fixed_to_float(raw), raw)


def compute_both_disparities(rect, num_disparities=None):
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
    valid = np.isfinite(disparity_px)
    out = np.zeros((*disparity_px.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    vals = disparity_px[valid]
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((disparity_px - lo) / (hi - lo), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0)
    coloured = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    coloured[~valid] = 0
    return coloured


# =============================================================================
# 5. DEPTH  Z = (f * B) / d
# =============================================================================
def metres_to_cm(metres: float | None) -> float | None:
    return None if metres is None else metres * 100.0


def disparity_to_depth(disparity: DisparityResult, calib: CalibrationResult) -> DepthResult:
    print("=" * 60)
    print("5. DEPTH ESTIMATION  (Z = f * B / d)")
    print("=" * 60)
    f, B, d = calib.focal_length_px, calib.baseline_m, disparity.disparity_px
    depth = np.full(d.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    depth[valid] = (f * B) / d[valid]
    depth[depth > 50.0] = np.nan
    depth[depth < 0.05] = np.nan
    print(f"  Using disparity from : {disparity.name}")
    print(f"  Focal length f (px)  : {f:.2f}")
    print(f"  Baseline B (m)       : {B:.4f}")
    print(f"  Valid depth pixels   : {int(np.isfinite(depth).sum())} / {depth.size}")
    if not calib.is_metric:
        print("  WARNING: experimental calibration - depths are ILLUSTRATIVE only.")
    return DepthResult(depth, f, B, calib.is_metric, disparity.name)


# =============================================================================
# 6-7. BOTTLE DETECTION + DEPTH
# =============================================================================
def detect_blue_bottle(image_bgr: np.ndarray) -> BottleDetection:
    print("=" * 60)
    print("6. BOTTLE DETECTION (HSV + morphology + contours)")
    print("=" * 60)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_BLUE_RANGES:
        mask = cv2.bitwise_or(
            mask, cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_BOTTLE_AREA_RATIO * mask.shape[0] * mask.shape[1]
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / float(max(w, 1))
        width_penalty = 1.0 if w < 0.55 * mask.shape[1] else 0.15
        score = area * min(aspect, 4.0) * width_penalty
        candidates.append((score, area, cnt, (x, y, w, h)))

    if not candidates:
        print("  Bottle NOT detected - adjust HSV ranges if needed.")
        return BottleDetection(False, mask, None, None, None, 0)

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, best_area, best, (x, y, w, h) = candidates[0]
    M = cv2.moments(best)
    if M["m00"] > 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [best], -1, 255, thickness=-1)
    print(f"  Bottle found: bbox=(x={x}, y={y}, w={w}, h={h}), "
          f"centroid=({cx}, {cy}), area={int(best_area)} px")
    return BottleDetection(True, clean, best, (x, y, w, h), (cx, cy), int(best_area))


def draw_bottle_overlay(image_bgr, det: BottleDetection) -> np.ndarray:
    out = image_bgr.copy()
    if not det.found or det.bbox is None:
        cv2.putText(out, "Bottle not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return out
    x, y, w, h = det.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if det.contour is not None:
        cv2.drawContours(out, [det.contour], -1, (0, 255, 255), 2)
    if det.centroid is not None:
        cv2.circle(out, det.centroid, 6, (0, 0, 255), -1)
        cv2.putText(out, "Bottle", (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def bottle_depth_stats(det, disparity, depth) -> BottleDepthStats:
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
    centre_disp = centre_depth = None
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
            local_mask & np.isfinite(local_disp)
            & (local_disp >= d_lo) & (local_disp <= d_hi)
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
    if median_depth is not None:
        print(f"  Median depth                     : "
              f"{median_depth} m  ({metres_to_cm(median_depth)} cm)")
    else:
        print("  Median depth: unavailable")
    if centre_depth is not None:
        print(f"  Centre depth                     : "
              f"{centre_depth} m  ({metres_to_cm(centre_depth)} cm)")
    else:
        print("  Centre depth: unavailable")
    if not depth.is_metric:
        print("  NOTE: values inherit EXPERIMENTAL calibration - not ground truth.")

    return BottleDepthStats(
        median_disp, centre_disp, median_depth, centre_depth, int(disp_vals.size)
    )


# =============================================================================
# 8. VISUALISATION
# =============================================================================
def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_fig(fig, path, show):
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_cv_image(path, bgr):
    cv2.imwrite(str(path), bgr)
    print(f"  Saved: {path.name}")


def visualise_original(pair, out_dir, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].imshow(_bgr_to_rgb(pair.left_bgr)); axes[0].set_title("Left image"); axes[0].axis("off")
    axes[1].imshow(_bgr_to_rgb(pair.right_bgr)); axes[1].set_title("Right image"); axes[1].axis("off")
    fig.suptitle("Original Stereo Pair")
    _save_fig(fig, out_dir / "01_original_stereo.png", show)
    save_cv_image(out_dir / "01_left.png", pair.left_bgr)
    save_cv_image(out_dir / "01_right.png", pair.right_bgr)


def visualise_rectified(rect, out_dir, show=False):
    guides = draw_horizontal_guides(rect.left_rect_bgr, rect.right_rect_bgr)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(_bgr_to_rgb(guides))
    ax.set_title(f"Rectified Stereo Pair with Epipolar Guides\n({rect.method})")
    ax.axis("off")
    _save_fig(fig, out_dir / "02_rectified_stereo.png", show)
    save_cv_image(out_dir / "02_rectified_guides.png", guides)
    save_cv_image(out_dir / "02_left_rectified.png", rect.left_rect_bgr)
    save_cv_image(out_dir / "02_right_rectified.png", rect.right_rect_bgr)


def visualise_disparity(bm, sgbm, out_dir, show=False):
    bm_color = colourise_disparity(bm.disparity_px)
    sgbm_color = colourise_disparity(sgbm.disparity_px)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(_bgr_to_rgb(bm_color)); axes[0].set_title("StereoBM disparity"); axes[0].axis("off")
    axes[1].imshow(_bgr_to_rgb(sgbm_color)); axes[1].set_title("StereoSGBM disparity"); axes[1].axis("off")
    fig.suptitle("Disparity Map Comparison")
    _save_fig(fig, out_dir / "03_disparity_comparison.png", show)

    def _grey(d):
        valid = np.isfinite(d)
        g = np.zeros(d.shape, dtype=np.uint8)
        if np.any(valid):
            vals = d[valid]
            lo, hi = np.percentile(vals, [2, 98])
            scaled = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            g = (np.nan_to_num(scaled, nan=0.0) * 255).astype(np.uint8)
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


def visualise_bottle(image_bgr, det, out_dir, show=False):
    overlay = draw_bottle_overlay(image_bgr, det)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(_bgr_to_rgb(overlay)); axes[0].set_title("Bottle detection + bounding box"); axes[0].axis("off")
    axes[1].imshow(det.mask, cmap="gray"); axes[1].set_title("Segmentation mask"); axes[1].axis("off")
    fig.suptitle("Blue Bottle Detection")
    _save_fig(fig, out_dir / "05_bottle_detection.png", show)
    save_cv_image(out_dir / "05_bottle_overlay.png", overlay)
    save_cv_image(out_dir / "05_bottle_mask.png", det.mask)
    return overlay


def visualise_depth(depth, det, stats, overlay_bgr, out_dir, show=False):
    depth_m = depth.depth_m
    valid = np.isfinite(depth_m)
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

    bottle_depth_vis = overlay_bgr.copy()
    if det.found and det.mask is not None:
        m = det.mask > 0
        bottle_depth_vis[m] = cv2.addWeighted(overlay_bgr, 0.35, colour, 0.65, 0)[m]
        label_y = 30
        if stats.median_depth_m is not None:
            txt = (f"Median Z ~ {stats.median_depth_m:.2f} m "
                   f"({metres_to_cm(stats.median_depth_m):.1f} cm)")
            cv2.putText(bottle_depth_vis, txt, (10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            label_y += 24
        if stats.centre_depth_m is not None:
            txt = (f"Centre Z ~ {stats.centre_depth_m:.2f} m "
                   f"({metres_to_cm(stats.centre_depth_m):.1f} cm)")
            cv2.putText(bottle_depth_vis, txt, (10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if not depth.is_metric:
            cv2.putText(bottle_depth_vis, "EXPERIMENTAL DEPTH", (10, label_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    fig2, ax2 = plt.subplots(figsize=(6, 7))
    ax2.imshow(_bgr_to_rgb(bottle_depth_vis))
    ax2.set_title("Bottle-specific Depth Visualisation")
    ax2.axis("off")
    _save_fig(fig2, out_dir / "07_bottle_depth.png", show)
    save_cv_image(out_dir / "07_bottle_depth_overlay.png", bottle_depth_vis)


def write_results_txt(path, pair, calib, det, stats, depth):
    lines = [
        "Computer Vision Lab 2 - Stereo Vision & Depth Estimation",
        "=" * 64,
        f"Left image          : {pair.left_path.name}",
        f"Right image         : {pair.right_path.name}",
        f"Working resolution  : {pair.width} x {pair.height} (scale={pair.scale:.4f})",
        "",
        "Calibration",
        "-" * 64,
        f"Mode                : {calib.mode}",
        f"Focal length f (px) : {depth.focal_px:.4f}",
        f"Baseline B (m)      : {depth.baseline_m:.6f}",
        f"Metric depths?      : {depth.is_metric}",
    ]
    for note in calib.notes:
        lines.append(f"NOTE: {note}")
    lines += ["", "Bottle detection", "-" * 64, f"Detected            : {det.found}"]
    if det.found and det.bbox is not None and det.centroid is not None:
        x, y, w, h = det.bbox
        cx, cy = det.centroid
        lines += [
            f"Bounding box (x,y,w,h): ({x}, {y}, {w}, {h})",
            f"Centroid (x, y)       : ({cx}, {cy})",
            f"Contour area (px)     : {det.area_px}",
        ]
    lines += [
        "", "Bottle depth", "-" * 64,
        f"Disparity source      : {depth.source_disparity}",
        f"Valid bottle pixels   : {stats.valid_pixels}",
        f"Median disparity (px) : {stats.median_disparity_px}",
        f"Centre disparity (px) : {stats.centre_disparity_px}",
    ]
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
    lines += ["", "Equation used:  Z = (f * B) / d"]
    if not depth.is_metric:
        lines.append(
            "WARNING: Experimental calibration - do NOT treat these depths "
            "as ground-truth metric measurements."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {path.name}")


def print_final_summary(det, stats, depth, calib):
    print("=" * 60)
    print("9. RESULTS SUMMARY")
    print("=" * 60)
    if det.found and det.bbox is not None and det.centroid is not None:
        x, y, w, h = det.bbox
        print(f"  Bottle bbox (x,y,w,h) : ({x}, {y}, {w}, {h})")
        print(f"  Bottle centroid       : {det.centroid}")
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


# =============================================================================
# PIPELINE
# =============================================================================
def run_experiment(
    left_path=None,
    right_path=None,
    output_dir=None,
    max_long_edge=MAX_LONG_EDGE,
    show=False,
) -> dict:
    left_path = Path(left_path or LEFT_IMAGE)
    right_path = Path(right_path or RIGHT_IMAGE)
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("#  CV Lab 2 - Stereo Vision & Depth Estimation")
    print("#" * 60 + "\n")

    pair = load_stereo_pair(left_path, right_path, max_long_edge=max_long_edge)
    calib = obtain_calibration((pair.width, pair.height))
    rect = rectify_stereo(pair, calib)
    num_disp = suggest_num_disparities(rect, fallback=STEREO_SGBM["numDisparities"])
    bm, sgbm = compute_both_disparities(rect, num_disparities=num_disp)
    depth = disparity_to_depth(sgbm, calib)

    det = detect_blue_bottle(rect.left_rect_bgr)
    if not det.found:
        print("  Retrying bottle detection on the original left image...")
        det = detect_blue_bottle(pair.left_bgr)

    stats = bottle_depth_stats(det, sgbm, depth)

    print("=" * 60)
    print("8. VISUALISATIONS")
    print("=" * 60)
    visualise_original(pair, output_dir, show=show)
    visualise_rectified(rect, output_dir, show=show)
    visualise_disparity(bm, sgbm, output_dir, show=show)
    overlay = visualise_bottle(rect.left_rect_bgr, det, output_dir, show=show)
    visualise_depth(depth, det, stats, overlay, output_dir, show=show)

    print_final_summary(det, stats, depth, calib)
    write_results_txt(output_dir / "results.txt", pair, calib, det, stats, depth)

    print("=" * 60)
    print("10. OUTPUT")
    print("=" * 60)
    print(f"  All artefacts written to: {output_dir}")
    print("  Done.\n")
    return {
        "pair": pair, "calib": calib, "rect": rect, "bm": bm, "sgbm": sgbm,
        "depth": depth, "detection": det, "bottle_stats": stats,
        "output_dir": output_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="CV Lab 2: Stereo Vision and Depth Estimation"
    )
    parser.add_argument("--left", type=Path, default=LEFT_IMAGE)
    parser.add_argument("--right", type=Path, default=RIGHT_IMAGE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--full-res", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Using Python packages: OpenCV, NumPy, Matplotlib")
    print(f"Left : {args.left}")
    print(f"Right: {args.right}")
    print(f"Out  : {args.output}\n")
    run_experiment(
        left_path=args.left,
        right_path=args.right,
        output_dir=args.output,
        max_long_edge=None if args.full_res else MAX_LONG_EDGE,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
