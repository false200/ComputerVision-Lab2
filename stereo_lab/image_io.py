"""Stereo image loading, validation and optional resizing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StereoPair:
    """Container for a left/right image pair and shared metadata."""

    left_bgr: np.ndarray
    right_bgr: np.ndarray
    left_path: Path
    right_path: Path
    scale: float = 1.0  # applied resize scale (1.0 = native)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.left_bgr.shape

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


def load_stereo_pair(
    left_path: Path | str,
    right_path: Path | str,
    max_long_edge: int | None = None,
) -> StereoPair:
    """
    Load left and right images, verify compatibility, optionally resize.

    Raises
    ------
    FileNotFoundError
        If either path does not exist or OpenCV cannot decode the file.
    ValueError
        If the two images have incompatible shapes after loading.
    """
    left_path = Path(left_path)
    right_path = Path(right_path)

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
            f"Incompatible stereo pair shapes: left {left.shape} vs right {right.shape}. "
            "Both images must share height, width and channel count."
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
        else:
            print("  No resize needed (already within max_long_edge).")

    return StereoPair(
        left_bgr=left,
        right_bgr=right,
        left_path=left_path,
        right_path=right_path,
        scale=scale,
    )