"""
End-to-end Stereo Vision Lab 2 pipeline.

Orchestrates loading → calibration → rectification → disparity → depth →
bottle detection → visualisation → numerical summary.
"""

from __future__ import annotations

from pathlib import Path

from . import config
from .bottle_detection import bottle_depth_stats, detect_blue_bottle
from .calibration import obtain_calibration
from .depth import disparity_to_depth
from .disparity import compute_both_disparities
from .image_io import load_stereo_pair
from .rectification import rectify_stereo, suggest_num_disparities
from .visualization import (
    print_final_summary,
    visualise_bottle,
    visualise_depth,
    visualise_disparity,
    visualise_original,
    visualise_rectified,
    write_results_txt,
)


def run_experiment(
    left_path: Path | str | None = None,
    right_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    max_long_edge: int | None = config.MAX_LONG_EDGE,
    show: bool = False,
) -> dict:
    """
    Execute the full lab experiment and save all artefacts.

    Parameters
    ----------
    left_path, right_path
        Stereo image paths (default: project ``left.jpg`` / ``right.jpg``).
    output_dir
        Directory for figures and ``results.txt``.
    max_long_edge
        Optional downscale limit for interactive speed.
    show
        If True, display Matplotlib windows interactively.

    Returns
    -------
    dict
        Key intermediate results for programmatic inspection / tests.
    """
    left_path = Path(left_path or config.LEFT_IMAGE)
    right_path = Path(right_path or config.RIGHT_IMAGE)
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("#  CV Lab 2 - Stereo Vision & Depth Estimation")
    print("#" * 60 + "\n")

    # 1. Load
    pair = load_stereo_pair(left_path, right_path, max_long_edge=max_long_edge)

    # 2. Calibration (chessboard if available, else experimental)
    image_size = (pair.width, pair.height)
    calib = obtain_calibration(image_size)

    # Scale experimental / calibrated focal length if images were resized.
    # Chessboard path already used working image_size; experimental focal is
    # defined for the working resolution in config, so no extra scale needed
    # beyond what obtain_calibration already used.

    # 3. Rectification
    rect = rectify_stereo(pair, calib)

    # 4. Disparity (BM + SGBM) — numDisparities from observed parallax
    num_disp = suggest_num_disparities(rect, fallback=config.STEREO_SGBM["numDisparities"])
    bm, sgbm = compute_both_disparities(rect, num_disparities=num_disp)

    # 5. Depth from preferred SGBM disparity
    depth = disparity_to_depth(sgbm, calib)

    # 6. Bottle detection on rectified left image (falls back to original left
    #    if rectification destroyed the colour blob — uncommon but safe).
    det = detect_blue_bottle(rect.left_rect_bgr)
    if not det.found:
        print("  Retrying bottle detection on the original left image...")
        det = detect_blue_bottle(pair.left_bgr)

    # 7. Bottle depth statistics
    stats = bottle_depth_stats(det, sgbm, depth)

    # 8. Visualisations
    print("=" * 60)
    print("8. VISUALISATIONS")
    print("=" * 60)
    visualise_original(pair, output_dir, show=show)
    visualise_rectified(rect, output_dir, show=show)
    visualise_disparity(bm, sgbm, output_dir, show=show)
    overlay = visualise_bottle(rect.left_rect_bgr, det, output_dir, show=show)
    visualise_depth(depth, det, stats, overlay, output_dir, show=show)

    # 9 / 10. Summary + save numerical results
    print_final_summary(det, stats, depth, calib)
    write_results_txt(output_dir / "results.txt", pair, calib, det, stats, depth)

    print("=" * 60)
    print("10. OUTPUT")
    print("=" * 60)
    print(f"  All artefacts written to: {output_dir}")
    print("  Done.\n")

    return {
        "pair": pair,
        "calib": calib,
        "rect": rect,
        "bm": bm,
        "sgbm": sgbm,
        "depth": depth,
        "detection": det,
        "bottle_stats": stats,
        "output_dir": output_dir,
    }