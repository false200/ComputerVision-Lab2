# Computer Vision Lab 2 — Stereo Vision and Depth Estimation

Estimate depth from a left/right image pair using classical stereo vision only (**OpenCV**, **NumPy**, **Matplotlib**). No deep learning.

All code is in one short file: **`run_lab.py`**.

Core pipeline:

```text
load → rectify → disparity (StereoBM + StereoSGBM) → depth Z = (f × B) / d
```

Kept explicitly: **stereo pair**, **disparity maps**, and **depth**.

---

## Why this works

Two photos of the same scene from slightly different positions create **parallax**. Nearby objects shift more than far ones. That horizontal shift after alignment is **disparity** `d`.

```text
d = x_left − x_right
Z = (f × B) / d
```

| Symbol | Meaning |
|--------|---------|
| `d` | Disparity in pixels |
| `f` | Focal length in pixels |
| `B` | Baseline (metres) between camera centres |
| `Z` | Depth in metres |

Larger `d` → closer object. Smaller `d` → farther object.

---

## What the code does

### 1. Load
Reads `left.jpg` and `right.jpg`, checks matching sizes, optionally resizes for speed.

### 2. Rectify
Uses ORB feature matches to estimate the vertical offset between views, then shifts the right image so corresponding points share the same row. If the disparity trend is negative, left/right are swapped so OpenCV gets positive disparities. Green guide lines are drawn for visual checks.

### 3. Disparity
- **StereoBM** — local block matching (fast, noisier)
- **StereoSGBM** — semi-global matching (cleaner; used for depth)

### 4. Depth
Applies `Z = (f * B) / d` on the SGBM map. Invalid / zero disparities are skipped (no divide-by-zero).

`f` and `B` in this lab are **experimental placeholders** (`700` px and `0.08` m). Depths demonstrate the formula; they are not ground-truth metres without real camera calibration.

---

## Run

```powershell
& "$env:LocalAppData\Programs\Python\Python311\python.exe" -m pip install -r requirements.txt
& "$env:LocalAppData\Programs\Python\Python311\python.exe" run_lab.py
```

```bash
python run_lab.py --show
python run_lab.py --full-res
```

---

## Outputs (`outputs/`)

| File | Content |
|------|---------|
| `01_stereo_pair.png` | Original left/right |
| `02_rectified.png` | Rectified pair + guides |
| `03_disparity_bm.png` | StereoBM disparity (colour) |
| `03_disparity_bm_gray.png` | StereoBM disparity (grey) |
| `04_disparity_sgbm.png` | StereoSGBM disparity (colour) |
| `04_disparity_sgbm_gray.png` | StereoSGBM disparity (grey) |
| `05_disparity_comparison.png` | BM vs SGBM side by side |
| `06_depth.png` | Depth map from SGBM |
| `results.txt` | `f`, `B`, both median disparities, median depth |

---

## Project layout

```text
left.jpg / right.jpg
run_lab.py
requirements.txt
README.md
outputs/          # created on run (gitignored)
```

---

## Formula reminder

```text
Z = (f * B) / d
```

Closer → larger `d` → smaller `Z`.  
Farther → smaller `d` → larger `Z`.
