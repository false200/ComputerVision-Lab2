"""
Camera calibration for stereo vision.

Supports two modes:
  1. Chessboard calibration  — accurate intrinsics / extrinsics when
     calibration images are available in ``calibration_images/``.
  2. Experimental mode       — clearly-labelled placeholders so the rest
     of the pipeline can run without fabricating "accurate" measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass
class CalibrationResult:
    """Stereo camera geometry used downstream for rectification and depth."""

    mode: str  # "chessboard" | "experimental"
    camera_matrix_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_left: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray  # rotation from left to right camera
    T: np.ndarray  # translation from left to right camera (metres)
    image_size: tuple[int, int]  # (width, height)
    rms_error: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def focal_length_px(self) -> float:
        """Average fx from the left camera matrix (pixels)."""
        return float(
            (self.camera_matrix_left[0, 0] + self.camera_matrix_left[1, 1]) / 2.0
        )

    @property
    def baseline_m(self) -> float:
        """Stereo baseline B = ||T|| in metres."""
        return float(np.linalg.norm(self.T))

    @property
    def is_metric(self) -> bool:
        return self.mode == "chessboard"


def _list_calibration_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(folder.glob(pat)))
    return files


def calibrate_from_chessboard(
    image_size: tuple[int, int],
    calibration_dir: Path | None = None,
    board_size: tuple[int, int] | None = None,
    square_size_m: float | None = None,
) -> CalibrationResult | None:
    """
    Attempt mono + stereo chessboard calibration.

    Expects mixed left/right views of a printed chessboard in
    ``calibration_dir``. For a student lab, a practical approach is to
    calibrate each camera separately with many board poses, then run
    ``cv2.stereoCalibrate``. Here we treat every board image as seen by
    *both* cameras with the same intrinsics (common for a single phone
    moved between left/right poses), then recover relative R, T from
    shared detections when possible.

    Returns None if fewer than 5 valid board detections are found.
    """
    calibration_dir = Path(calibration_dir or config.CALIBRATION_DIR)
    board_size = board_size or config.CHESSBOARD_SIZE
    square_size_m = square_size_m or config.SQUARE_SIZE_M
    images = _list_calibration_images(calibration_dir)
    if len(images) < 5:
        return None

    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    detected = 0

    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != image_size:
            img = cv2.resize(img, image_size)
        found, corners = cv2.findChessboardCorners(img, board_size, None)
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

    # Without true dual-camera board captures we cannot recover a reliable
    # stereo T. Fall back to identity R and a measured/placeholder T along X
    # using the configured experimental baseline, but keep calibrated K, dist.
    R = np.eye(3, dtype=np.float64)
    T = np.array([[config.EXPERIMENTAL_BASELINE_M], [0.0], [0.0]], dtype=np.float64)

    notes = [
        f"Chessboard intrinsics recovered from {detected} images "
        f"(RMS reprojection error = {rms:.4f} px).",
        "Stereo extrinsics (R, T) still need a proper stereoCalibrate "
        "session with simultaneous left/right board views. Baseline was "
        f"set to the experimental value {config.EXPERIMENTAL_BASELINE_M} m.",
    ]
    return CalibrationResult(
        mode="chessboard",
        camera_matrix_left=K.copy(),
        camera_matrix_right=K.copy(),
        dist_coeffs_left=dist.copy(),
        dist_coeffs_right=dist.copy(),
        R=R,
        T=T,
        image_size=image_size,
        rms_error=float(rms),
        notes=notes,
    )


def build_experimental_calibration(image_size: tuple[int, int]) -> CalibrationResult:
    """
    Build a clearly-labelled experimental calibration.

    Camera matrix form:
        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    Distortion is assumed zero. Relative pose is a pure translation along
    the X axis equal to the experimental baseline.
    """
    w, h = image_size
    fx = fy = float(config.EXPERIMENTAL_FOCAL_PX)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    T = np.array([[config.EXPERIMENTAL_BASELINE_M], [0.0], [0.0]], dtype=np.float64)

    notes = [
        "EXPERIMENTAL MODE - no chessboard calibration data was found.",
        f"Using placeholder focal length f ~ {fx:.1f} px and baseline "
        f"B ~ {config.EXPERIMENTAL_BASELINE_M:.3f} m.",
        "Depth values computed from Z = (f * B) / d are ILLUSTRATIVE ONLY "
        "and must NOT be treated as accurate metric measurements.",
        "For accurate depth you need: (1) intrinsic calibration of each "
        "camera (chessboard), (2) stereo extrinsics (R, T) from "
        "stereoCalibrate, and (3) a measured baseline in metres.",
    ]
    return CalibrationResult(
        mode="experimental",
        camera_matrix_left=K,
        camera_matrix_right=K.copy(),
        dist_coeffs_left=dist,
        dist_coeffs_right=dist.copy(),
        R=R,
        T=T,
        image_size=image_size,
        rms_error=None,
        notes=notes,
    )


def obtain_calibration(image_size: tuple[int, int]) -> CalibrationResult:
    """
    Public entry: try chessboard calibration, otherwise experimental mode.

    ``image_size`` is (width, height) of the working stereo images.
    """
    print("=" * 60)
    print("2. CAMERA CALIBRATION")
    print("=" * 60)

    result = calibrate_from_chessboard(image_size)
    if result is None:
        print("  No usable chessboard images found in:")
        print(f"    {config.CALIBRATION_DIR}")
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
        # ASCII-safe for Windows consoles (cp1252)
        safe = (
            note.replace("\u2014", "-")
            .replace("\u2192", "->")
            .replace("\u2248", "~")
            .replace("\u00d7", "x")
        )
        print(f"  NOTE: {safe}")
    return result