#!/usr/bin/env python3
"""
Fake-LR baseline, but on the TEST tiles this time (not training), so it lines up
tile-for-tile with the 128-tile model eval.

Fake-LR = take a PlanetScope HR test tile, bicubic down 192->60, then bicubic
back up 60->192. Same instrument, same day, same registration, so the ONLY thing
that changed is resolution. Score fake-LR against its own HR and you've isolated
the resolution/scene-content part with zero cross-sensor bias.

Same conventions as the training pipeline (superres_eval.py / make_fakelr_eval.py):
  - PSNR/SSIM: per-channel per-image min-max -> uint8, MATLAB-style 11x11
    Gaussian SSIM, across all 4 bands (NIR included).
  - HR files come as (H,W,C) or (C,H,W); we sniff out the band axis.
  - macOS "._*" resource-fork junk gets skipped.
  - 200 vs 192: if HR is the bigger one, top-left crop to 192. (Peak content
    correlation checked out at offset (0,0), not center.)

No LPIPS here on purpose -- keeps the deps light and matches how the structural
and perceptual passes were split elsewhere. (Adding a fake-LR LPIPS pass would
just be an RGB-only [0,1] run with the lpips lib if it's ever wanted.)

Usage: edit HR_DIR below to point at the test HR folder, then:

    pip install numpy scipy scikit-image tifffile   # if you don't have them
    python make_fakelr_testset.py

Drops two files next to this script:
    fakelr_testset_metrics.csv        # one row per tile, per-channel + mean
    fakelr_testset_summary.txt        # quick console-style recap
"""

import os
import glob
import numpy as np

# EDIT THIS -- the PlanetScope HR tiles for the TEST set. Needs to be the 128 HR
# tiles whose stems match the model-output filenames (AOI_* and tile_*). Usual suspects:
#   ~/Desktop/Super_resolution_project/Test/hr_200
#   ~/Desktop/Super_resolution_project/Training Data/hr_200   (if the test HR lives there)
HR_DIR = os.path.expanduser("~/Desktop/Super_resolution_project/Test/hr_200")

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fakelr_testset_metrics.csv")
OUT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fakelr_testset_summary.txt")

TARGET = 192          # working tile size
DOWN = 60             # the LR size in between (matches the 60x60 pipeline)
BANDS = ["R", "G", "B", "NIR"]

# --- IO ---
def load_tif(path):
    """Load a GeoTIFF as a float32 (H, W, C) array, C=4."""
    try:
        import tifffile
        arr = tifffile.imread(path)
    except Exception:
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read()  # (C, H, W)
    arr = np.asarray(arr).astype(np.float32)

    # get it to (H, W, C) -- the band axis is the length-4 (or just smallest) one
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3:
        # the size-4 axis is bands; make that the channel axis
        band_axes = [i for i, s in enumerate(arr.shape) if s == 4]
        if band_axes:
            ax = band_axes[0]
        else:
            ax = int(np.argmin(arr.shape))  # fallback: smallest axis is channels
        arr = np.moveaxis(arr, ax, -1)
    return arr

def crop_topleft(arr, size=TARGET):
    """Top-left crop to size x size if the tile's bigger (the 200-vs-192 case)."""
    h, w = arr.shape[:2]
    if h >= size and w >= size:
        return arr[:size, :size, ...]
    return arr  # smaller than target: leave as-is, resize handles it

# --- resize ---
def bicubic(arr_hw_c, out_size):
    """Bicubic-resize an (H,W,C) float array to (out_size, out_size, C)."""
    from skimage.transform import resize
    return resize(
        arr_hw_c, (out_size, out_size, arr_hw_c.shape[2]),
        order=3, mode="reflect", anti_aliasing=True, preserve_range=True
    ).astype(np.float32)

def make_fakelr(hr):
    """The 192 -> 60 -> 192 bicubic round-trip -- this is what makes it "LR"."""
    small = bicubic(hr, DOWN)
    back = bicubic(small, TARGET)
    return back

# --- metrics ---
def to_uint8_per_channel(x2d):
    """Per-image min-max one 2D channel into uint8 [0,255]."""
    lo, hi = float(x2d.min()), float(x2d.max())
    if hi <= lo:
        return np.zeros_like(x2d, dtype=np.uint8)
    y = (x2d - lo) / (hi - lo)
    return (y * 255.0 + 0.5).astype(np.uint8)

def psnr_uint8(a, b):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    return float(psnr(a, b, data_range=255))

def ssim_uint8(a, b):
    # the MATLAB recipe: 11x11 Gaussian window, sigma 1.5
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(a, b, data_range=255, gaussian_weights=True,
                      win_size=11, sigma=1.5, use_sample_covariance=False))

# --- main ---
def main():
    files = sorted(glob.glob(os.path.join(HR_DIR, "*.tif")))
    files = [f for f in files if not os.path.basename(f).startswith("._")]
    if not files:
        raise SystemExit(f"No .tif files found in {HR_DIR!r}. Edit HR_DIR.")

    print(f"Found {len(files)} HR test tiles in {HR_DIR}")
    rows = []
    for i, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            hr = crop_topleft(load_tif(path))
            if hr.shape[2] < 4:
                print(f"  [skip] {stem}: only {hr.shape[2]} band(s)")
                continue
            hr = hr[..., :4]
            fake = make_fakelr(hr)

            rec = {"stem": stem, "hr_path": path}
            psnrs, ssims = [], []
            for c, band in enumerate(BANDS):
                a = to_uint8_per_channel(hr[..., c])
                b = to_uint8_per_channel(fake[..., c])
                p = psnr_uint8(a, b)
                s = ssim_uint8(a, b)
                rec[f"psnr_{band}"] = p
                rec[f"ssim_{band}"] = s
                psnrs.append(p); ssims.append(s)
            rec["psnr_mean"] = float(np.mean(psnrs))
            rec["ssim_mean"] = float(np.mean(ssims))
            rows.append(rec)
        except Exception as e:
            print(f"  [error] {stem}: {e}")

        if i % 20 == 0:
            print(f"  ...{i}/{len(files)}")

    # out to CSV
    import csv
    cols = (["stem", "hr_path"]
            + [f"psnr_{b}" for b in BANDS]
            + [f"ssim_{b}" for b in BANDS]
            + ["psnr_mean", "ssim_mean"])
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # and a quick recap
    import statistics as st
    pm = [r["psnr_mean"] for r in rows]
    sm = [r["ssim_mean"] for r in rows]
    lines = [
        f"fake-LR test-set summary (n={len(rows)})",
        f"  PSNR mean : {st.mean(pm):.3f}  (median {st.median(pm):.3f})",
        f"  SSIM mean : {st.mean(sm):.3f}  (median {st.median(sm):.3f})",
        f"  wrote: {OUT_CSV}",
    ]
    txt = "\n".join(lines)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)

if __name__ == "__main__":
    main()
