#!/usr/bin/env python3
"""
Computer Vision Lab 2 — Stereo Vision and Depth Estimation
==========================================================

Run the complete experiment on the provided left/right bottle images:

    python run_lab.py

Useful options:

    python run_lab.py --show              # also pop up Matplotlib windows
    python run_lab.py --full-res          # skip downscaling
    python run_lab.py --left L.jpg --right R.jpg

Requires: opencv-python, numpy, matplotlib
(see requirements.txt)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when executed as a script.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stereo_lab import config
from stereo_lab.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CV Lab 2: Stereo Vision and Depth Estimation"
    )
    parser.add_argument(
        "--left", type=Path, default=config.LEFT_IMAGE,
        help="Path to left stereo image (default: left.jpg)",
    )
    parser.add_argument(
        "--right", type=Path, default=config.RIGHT_IMAGE,
        help="Path to right stereo image (default: right.jpg)",
    )
    parser.add_argument(
        "--output", type=Path, default=config.OUTPUT_DIR,
        help="Directory for saved figures and results.txt",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display Matplotlib figures interactively",
    )
    parser.add_argument(
        "--full-res", action="store_true",
        help="Process at native resolution (slower)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_long_edge = None if args.full_res else config.MAX_LONG_EDGE

    print("Using Python packages: OpenCV, NumPy, Matplotlib")
    print(f"Left : {args.left}")
    print(f"Right: {args.right}")
    print(f"Out  : {args.output}")
    print()

    run_experiment(
        left_path=args.left,
        right_path=args.right,
        output_dir=args.output,
        max_long_edge=max_long_edge,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())