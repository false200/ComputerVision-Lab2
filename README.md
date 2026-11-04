# Computer Vision Lab 2 — Stereo Vision, Disparity, and Depth

Estimate depth from a left/right image pair using classical stereo vision only (**OpenCV**, **NumPy**, **Matplotlib**). No deep learning.

All code is in **`run_lab.py`**.

```text
load → rectify → correspondence / disparity (StereoBM + StereoSGBM) → depth
```

---

## 1. Pipeline overview

| Step | What happens | Formula / idea |
|------|----------------|----------------|
| Load | Read `left.jpg`, `right.jpg` | Same resolution required |
| Rectify | Align rows (epipolar lines horizontal) | Correspondence becomes 1-D |
| Disparity | Solve the **correspondence problem** | `d = u_l − u_r` |
| Depth | Perspective stereo equation | `Z = (f × B) / d` |

---

## 2. Formulas (full set used in this lab)

### 2.1 Perspective projection (pinhole camera)

A 3-D point `(x, y, z)` projects to the image as:

```text
u = fx * (x / z) + ox
v = fy * (y / z) + oy
```

For a **stereo** pair with baseline `b` along x (right camera shifted by `b`):

```text
Left:   (u_l, v_l) = ( fx * (x / z)       + ox ,  fy * (y / z) + oy )
Right:  (u_r, v_r) = ( fx * ((x - b) / z) + ox ,  fy * (y / z) + oy )
```

After rectification, `v_l = v_r` (same row). Only the horizontal coordinate differs.

### 2.2 Disparity from perspective projection

```text
d = u_l - u_r
  = fx*(x/z) - fx*((x - b)/z)
  = fx * (b / z)
```

So:

```text
d = (fx * b) / z
z = (fx * b) / d
```

In code we write this as:

```text
Z = (f * B) / d
```

with `f ≈ fx` (pixels) and `B = b` (metres).

### 2.3 Correspondence problem

**Problem:** given a pixel in the left image, find the matching pixel in the right image (same 3-D world point).

**In this lab:** StereoBM and StereoSGBM solve correspondence densely along each row after rectification. The match shift is disparity `d`.

ORB matching in `rectify()` also solves correspondence sparsely, only to estimate the vertical offset.

### 2.4 Symbol table

| Symbol | Meaning | In `run_lab.py` |
|--------|---------|-----------------|
| `(x,y,z)` | 3-D point in camera frame | recovered depth is `z` / `Z` |
| `(u_l, v_l)` | left image pixel | left rectified image |
| `(u_r, v_r)` | right image pixel | right rectified image |
| `fx`, `fy` | focal length (pixels) | `FOCAL_PX` (as `f`) |
| `ox`, `oy` | principal point | assumed near image centre |
| `b` / `B` | baseline (metres) | `BASELINE_M` |
| `d` | disparity (pixels) | StereoBM / StereoSGBM output |
| `Z` | depth (metres) | `disparity_to_depth()` |

### 2.5 Intuition

```text
Closer object  →  larger d  →  smaller Z
Farther object →  smaller d →  larger Z
```

---

## 3. Where each formula appears in code

| Concept | Location in `run_lab.py` |
|---------|---------------------------|
| `f`, `B` | `FOCAL_PX`, `BASELINE_M` at top |
| Correspondence / disparity | `compute_bm()`, `compute_sgbm()` |
| `d = u_l - u_r` | OpenCV stereo matchers (internally) |
| `Z = (f * B) / d` | `disparity_to_depth()` → `depth[valid] = (f * B) / disp[valid]` |
| Sparse correspondence | `match_offsets()` / ORB inside `rectify()` |

---

## 4. What the code does (steps)

### Load
Reads left/right, checks shapes match, optional resize for speed.

### Rectify
Estimates median vertical offset from ORB matches, shifts the right image so matches share a row. Swaps left/right if disparity would be negative. Draws green horizontal guides.

### Disparity (correspondence)
- **StereoBM** — block matching  
- **StereoSGBM** — semi-global matching (used for final depth)

### Depth
Applies perspective stereo formula `Z = (f * B) / d` on the SGBM disparity map. Skips invalid / zero `d`.

`f` and `B` are **experimental placeholders** (`700` px, `0.08` m). Depths illustrate the equation; they are not ground-truth metres without real calibration.

---

## 5. Run

```powershell
& "$env:LocalAppData\Programs\Python\Python311\python.exe" -m pip install -r requirements.txt
& "$env:LocalAppData\Programs\Python\Python311\python.exe" run_lab.py
```

```bash
python run_lab.py --show
python run_lab.py --full-res
```

---

## 6. Outputs (`outputs/`)

| File | Content |
|------|---------|
| `01_stereo_pair.png` | Original left/right |
| `02_rectified.png` | Rectified pair + guides |
| `03_disparity_bm.png` | StereoBM disparity (colour) |
| `03_disparity_bm_gray.png` | Gray map: **brighter = closer**, darker = farther |
| `04_disparity_sgbm.png` | StereoSGBM disparity (colour) |
| `04_disparity_sgbm_gray.png` | Gray map: **brighter = closer**, darker = farther |
| `05_disparity_comparison.png` | BM vs SGBM (colour) |
| `05b_disparity_gray_comparison.png` | BM vs SGBM (gray, bright=close) |
| `06_depth.png` | Depth from SGBM via `Z=(f*B)/d` |
| `results.txt` | `f`, `B`, disparities, depth |

---

## 7. Project layout

```text
left.jpg / right.jpg
run_lab.py
requirements.txt
README.md
outputs/          # created on run (gitignored)
```

---

## 8. Quick formula card

```text
Perspective (stereo):
  u_l = fx * (x / z) + ox
  u_r = fx * ((x - b) / z) + ox

Disparity:
  d = u_l - u_r = (fx * b) / z

Depth:
  z = (fx * b) / d
  Z = (f * B) / d          ← used in code
```
