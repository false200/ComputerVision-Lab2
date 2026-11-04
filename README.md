# Computer Vision Lab 2 — Simple Stereo | Camera Calibration

Classical stereo only (**OpenCV**, **NumPy**, **Matplotlib**). One file: **`run_lab.py`**.

This lab follows the lecture slides on **Simple Stereo**, **Depth & Disparity**, **Stereo Matching**, **Window-Based Methods**, and **SAD / SSD / NCC**.

```text
load → rectify (scan lines) → window matching (SAD/SSD/NCC) → disparity → (x,y,z)
```

---

## 1. Perspective projection (slide)

Left camera at origin, right camera at `(b, 0, 0)`:

```text
(u_l, v_l) = ( fx * (x / z) + ox ,  fy * (y / z) + oy )
(u_r, v_r) = ( fx * ((x - b) / z) + ox ,  fy * (y / z) + oy )
```

After **rectification**, `v_l = v_r` — matching is only along the **same horizontal scan line**.

---

## 2. Depth and disparity (slide)

```text
d = u_l - u_r                              # disparity
z = (b * fx) / d                           # depth (lecture highlight)
x = b * (u_l - ox) / d
y = b * fx * (v_l - oy) / (fy * d)
```

- Depth `z` is **inversely proportional** to disparity  
- Disparity is **proportional** to baseline `b`

In code: `disparity_to_xyz()` in `run_lab.py`.

---

## 3. Stereo matching / disparity map (slide)

**Goal:** find disparity between left and right pairs.

Gray encoding (like the ground-truth slide):

- **Brighter** → larger `d` → **closer**
- **Darker** → smaller `d` → **farther**

Your phone photos will not look as clean as the Tsukuba lab ground-truth map (that needs a calibrated stereo rig). We still use the same encoding and pipeline.

---

## 4. Window-based methods (slide)

1. Take a **template window `T`** around a pixel in the left image `E_l`  
2. Search along the **scan line `L`** in the right image `E_r` (same row)  
3. Choose the shift (disparity) with the best similarity score  

Dense version: **StereoBM** does this for every pixel using **SAD**.

Demo figure: `02b_window_scanline_demo.png` (red = `T`, white = `L`, green = match).

---

## 5. Similarity metrics (slide)

```text
SAD(k,l) = Σ |E_l(i,j) − E_r(i+k, j+l)|          # minimise
SSD(k,l) = Σ |E_l(i,j) − E_r(i+k, j+l)|^2         # minimise
NCC(k,l) = Σ E_l E_r / sqrt(Σ E_l² Σ E_r²)        # maximise
```

Implemented as `sad()`, `ssd()`, `ncc()` and printed for a centre template in `results.txt`.

---

## 6. How large should the window be? (slide)

| Window | Effect | In code |
|--------|--------|---------|
| Small (5 px) | Sensitive to noise | `WINDOW_SMALL = 5` |
| Large (~30 px) | Less noise, poorer edge localization | `WINDOW_LARGE = 31` |
| **Adaptive** | Try several sizes; fuse robustly | `WINDOW_ADAPTIVE = (5, 15, 31)` |

Comparison figure: `03_window_small_vs_large.png`.

---

## 7. Run

```powershell
& "$env:LocalAppData\Programs\Python\Python311\python.exe" -m pip install -r requirements.txt
& "$env:LocalAppData\Programs\Python\Python311\python.exe" run_lab.py
```

```bash
python run_lab.py --show
python run_lab.py --full-res
```

---

## 8. Outputs

| File | Lecture meaning |
|------|-----------------|
| `01_stereo_pair.png` | Left/right camera images |
| `02_rectified_scanlines.png` | Same-row epipolar / scan lines |
| `02b_window_scanline_demo.png` | Template `T` + search line `L` |
| `03_window_small_vs_large.png` | Window 5 vs 31 comparison |
| `04_disparity_adaptive_gray.png` | Disparity map (bright=close) |
| `04b_disparity_sgbm_gray.png` | SGBM disparity (bright=close) |
| `05_disparity_color.png` | Colour disparity |
| `06_depth.png` | Depth `z = (b·fx)/d` |
| `results.txt` | All formulas + SAD/SSD/NCC demo + numbers |

---

## 9. Honest note on “ground truth” maps

The clean lamp/bust disparity slide is **lab ground truth** (calibrated stereo, controlled scene).  
Our inputs are hand-held phone photos with **approximate** `fx` and `b` (no chessboard).  

We implement the **same maths and window-matching ideas**. The map will be noisier than Tsukuba GT until you add real chessboard calibration.
