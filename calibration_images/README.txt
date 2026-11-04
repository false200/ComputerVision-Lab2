Place chessboard calibration images here (JPG/PNG).

Expected board: 9x6 inner corners (change CHESSBOARD_SIZE in stereo_lab/config.py
if your board differs). Set SQUARE_SIZE_M to the real square size in metres.

When ≥ 5 board images with detectable corners are present, run_lab.py will
recover camera intrinsics automatically. For a true stereo baseline you still
need simultaneous left/right board captures and cv2.stereoCalibrate.

If this folder is empty, the lab runs in EXPERIMENTAL mode with clearly
labelled placeholder focal length and baseline — depths are illustrative only.
