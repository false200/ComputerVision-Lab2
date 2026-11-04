"""
Stereo rectification.

Strategies (tried in order for experimental / hand-held pairs):
  1. Essential-matrix pose recovery + cv2.stereoRectify  (preferred)
  2. Uncalibrated homographies via stereoRectifyUncalibrated
  3. Gentle vertical translation from median feature offset (last resort)

For chessboard calibration mode, calibrated stereoRectify + remap is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CalibrationResult
from .image_io import StereoPair


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


def _orb_matches(left_g: np.ndarray, right_g: np.ndarray, keep: int = 250):
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


def _dx_dy_stats(pts1: np.ndarray, pts2: np.ndarray) -> dict:
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
        left_rect_bgr=left_bgr,
        right_rect_bgr=right_bgr,
        left_rect_gray=cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY),
        right_rect_gray=cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY),
        method=method,
        **kwargs,
    )


def _rectify_essential(
    pair: StereoPair, calib: CalibrationResult, pts1: np.ndarray, pts2: np.ndarray
) -> RectificationResult | None:
    """Recover R,T from the essential matrix, then apply stereoRectify."""
    K = calib.camera_matrix_left
    dist = calib.dist_coeffs_left
    w, h = pair.width, pair.height

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None

    inl = mask.ravel().astype(bool)
    if inl.sum() < 20:
        return None

    _, R, T, pose_mask = cv2.recoverPose(E, pts1[inl], pts2[inl], K)
    # Normalise translation to the configured baseline length (metres).
    t_norm = np.linalg.norm(T)
    if t_norm < 1e-8:
        return None
    T = T / t_norm * calib.baseline_m

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist, K, dist, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.9,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)
    left_rect = cv2.remap(pair.left_bgr, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(pair.right_bgr, map2x, map2y, cv2.INTER_LINEAR)

    # Sanity: reject pathological warps that destroy most of the image.
    valid_frac = float(np.mean(cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY) > 0))
    if valid_frac < 0.55:
        print(f"  Essential rectification rejected (valid fraction={valid_frac:.2f}).")
        return None

    return _pack(
        left_rect, right_rect,
        method="essential matrix + stereoRectify",
        Q=Q,
    )


def _rectify_uncalibrated(
    pair: StereoPair, pts1: np.ndarray, pts2: np.ndarray
) -> RectificationResult | None:
    """Homography rectification; rejected if the warp is too aggressive."""
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
    # Also reject if the bottle region would be heavily sheared: check corner
    # displacement magnitude.
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
        method="uncalibrated (ORB + F + stereoRectifyUncalibrated)",
        H1=H1, H2=H2,
    )


def _rectify_vertical_align(
    pair: StereoPair, pts1: np.ndarray, pts2: np.ndarray
) -> RectificationResult:
    """
    Gentle rectification for nearly-aligned hand-held pairs.

    Applies a pure vertical translation to the right image so that the median
    feature y-offset becomes zero. Horizontal parallax (disparity) is preserved.
    """
    stats = _dx_dy_stats(pts1, pts2)
    dy = int(round(stats["dy_median"]))
    print(f"  Applying vertical alignment shift of {dy} px to the right image.")
    print(
        f"  Pre-align disparity stats: dx median={stats['dx_median']:.1f} "
        f"[{stats['dx_p10']:.1f}, {stats['dx_p90']:.1f}], "
        f"dy median={stats['dy_median']:.1f} (std={stats['dy_std']:.1f})"
    )

    h, w = pair.height, pair.width
    M = np.float32([[1, 0, 0], [0, 1, dy]])  # shift right image DOWN by dy
    # If y_left - y_right = +dy_med, right points are ABOVE left → shift right down.
    right_rect = cv2.warpAffine(
        pair.right_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    left_rect = pair.left_bgr.copy()
    return _pack(
        left_rect, right_rect,
        method=f"vertical translation alignment (dy={dy}px)",
        match_dx_stats=stats,
    )


def _rectify_calibrated(
    pair: StereoPair, calib: CalibrationResult
) -> RectificationResult:
    w, h = pair.width, pair.height
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        calib.camera_matrix_left,
        calib.dist_coeffs_left,
        calib.camera_matrix_right,
        calib.dist_coeffs_right,
        (w, h),
        calib.R,
        calib.T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.9,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib.camera_matrix_left, calib.dist_coeffs_left, R1, P1, (w, h), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib.camera_matrix_right, calib.dist_coeffs_right, R2, P2, (w, h), cv2.CV_32FC1
    )
    left_rect = cv2.remap(pair.left_bgr, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(pair.right_bgr, map2x, map2y, cv2.INTER_LINEAR)
    return _pack(left_rect, right_rect, method="calibrated (stereoRectify + remap)", Q=Q)


def rectify_stereo(
    pair: StereoPair, calib: CalibrationResult
) -> RectificationResult:
    """Rectify so corresponding points share the same image row."""
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

    result = _rectify_essential(pair, calib, pts1, pts2)
    if result is None:
        result = _rectify_uncalibrated(pair, pts1, pts2)
    if result is None:
        result = _rectify_vertical_align(pair, pts1, pts2)

    result.match_dx_stats = stats
    print(f"  Method: {result.method}")
    print("  Corresponding points should now lie on roughly the same horizontal rows.")
    return result


def draw_horizontal_guides(
    left_bgr: np.ndarray, right_bgr: np.ndarray, num_lines: int = 12
) -> np.ndarray:
    """Side-by-side rectified pair with evenly spaced horizontal guide lines."""
    h = left_bgr.shape[0]
    canvas = np.hstack([left_bgr, right_bgr]).copy()
    for y in np.linspace(0, h - 1, num_lines, dtype=int):
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (0, 255, 0), 1)
    return canvas


def suggest_num_disparities(rect: RectificationResult, fallback: int = 64) -> int:
    """
    Pick a numDisparities (multiple of 16) from observed horizontal match offsets.
    """
    stats = rect.match_dx_stats
    if not stats:
        return fallback
    # Need headroom above the 90th percentile disparity.
    need = abs(stats["dx_p90"]) + 16
    num = int(np.ceil(need / 16.0) * 16)
    return int(np.clip(num, 16, 160))
