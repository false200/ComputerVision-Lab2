"""
Central configuration for the stereo vision lab.

IMPORTANT — Calibration honesty:
  No chessboard / factory calibration is shipped with this dataset.
  Experimental mode therefore uses *placeholder* focal length and baseline
  so the pipeline can run end-to-end. Those values are NOT ground truth.
  Metric depths reported in experimental mode are illustrative only.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEFT_IMAGE = PROJECT_ROOT / "left.jpg"
RIGHT_IMAGE = PROJECT_ROOT / "right.jpg"
CALIBRATION_DIR = PROJECT_ROOT / "calibration_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
# Downscale long edge for interactive speed while keeping enough detail
# for block matching. Set to None to process at native resolution.
MAX_LONG_EDGE = 800

# ---------------------------------------------------------------------------
# Chessboard calibration (used only if images exist in CALIBRATION_DIR)
# ---------------------------------------------------------------------------
CHESSBOARD_SIZE = (9, 6)  # inner corners (cols, rows)
SQUARE_SIZE_M = 0.025     # physical square size in metres (edit to match board)

# ---------------------------------------------------------------------------
# Experimental stereo geometry (USED ONLY WHEN CALIBRATION IS UNAVAILABLE)
# ---------------------------------------------------------------------------
# These are deliberate approximations for a phone-held stereo pair so that
# Z = (f * B) / d can be demonstrated. Replace with calibrated values for
# accurate metric depth.
#
#   EXPERIMENTAL_FOCAL_PX  — approximate focal length in pixels
#                            (≈ image_width / (2 * tan(FOV/2)); phone FOV ~60–70°)
#   EXPERIMENTAL_BASELINE_M — approximate left–right camera separation (metres)
#
EXPERIMENTAL_FOCAL_PX = 700.0
# Placeholder only. Feature matching on this pair shows ~9-15 px of horizontal
# parallax; an 8 cm baseline would imply a multi-metre depth, which may not
# match the true capture geometry. Measure B (or calibrate) for real metres.
EXPERIMENTAL_BASELINE_M = 0.08

# ---------------------------------------------------------------------------
# Stereo matching parameters (tuned for the provided bottle images)
# ---------------------------------------------------------------------------
# numDisparities must be divisible by 16.
# numDisparities is auto-tuned from feature matches when possible; these
# values are safe defaults for the provided bottle pair (small parallax).
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
    "P1": 8 * 3 * 7 * 7,       # 8 * channels * blockSize^2
    "P2": 32 * 3 * 7 * 7,      # 32 * channels * blockSize^2
    "disp12MaxDiff": 1,
    "uniquenessRatio": 12,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "preFilterCap": 63,
    "mode": 0,  # overridden to HH in disparity module when available
}

# ---------------------------------------------------------------------------
# Blue-bottle HSV segmentation (covers cyan upper body + navy lower body)
# ---------------------------------------------------------------------------
# OpenCV hue is 0–179. Blue / cyan typically sits around 90–130.
HSV_BLUE_RANGES = [
    # Light cyan / sky-blue (upper bottle)
    ((85, 60, 80), (115, 255, 255)),
    # Deep navy (lower bottle + cap)
    ((100, 40, 20), (130, 255, 180)),
]

MORPH_KERNEL_SIZE = 7
MIN_BOTTLE_AREA_RATIO = 0.005  # contour must cover ≥ 0.5 % of the image