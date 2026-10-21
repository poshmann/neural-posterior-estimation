#!/usr/bin/env python3
"""
Part A -- the new perceptual + structural metrics on the actual SR results.

For every model-variant folder under "Model Results/", match each output tile to
its PlanetScope HR twin in the test set and score the pair with:

    PSNR, SSIM   -- structural, per channel (our usual: per-image min-max ->
                    uint8, MATLAB-style 11x11 Gaussian SSIM)
    CW-SSIM      -- structural, shift-tolerant (Complex Wavelet SSIM, pyrtools)
    MS-SSIM      -- multi-scale SSIM (piq); the closest maintained cousin of CW-SSIM
    DISTS        -- perceptual, VGG-based (piq). lower = better
    CLIP-cos     -- perceptual, cosine distance of CLIP ViT-B/32 embeddings. lower = better
    LPIPS        -- perceptual, AlexNet. recomputed here so it's one consistent pass. lower = better

Conventions (hashed out with Lucas):
  * the network metrics (DISTS, CLIP) and the SSIM family run on RGB only, NIR
    dropped -- matching the LPIPS convention already in use, so every perceptual
    number stays comparable.
  * CW-SSIM ALSO gets a per-channel breakdown with NIR, since it's a classical
    metric with no pretrained net -- all 4 bands mean something there. Same shape
    as how PSNR/SSIM already do per-channel + total.
  * channel order R=0, G=1, B=2, NIR=3.
  * axis order sniffed automatically: HR tiles are (H,W,C), SR often (C,H,W).
  * macOS '._' junk filtered everywhere.

Runs on the M5 MacBook (MPS). Won't run in the sandbox -- DISTS/CLIP weights
download from the net and MPS isn't there.

Usage:
    python evaluate_new_metrics.py \
        --hr-dir   ".../Test/hr_200" \
        --results  ".../Model Results" \
        --out      ".../new_metrics" \
        --variants diffusion_sr_2000steps_no_adapt consistency_sr_1step_no_adapt_8xensemble \
        --limit 10          # PoC: 10 tiles per variant. drop it for the full 129.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.signal import convolve2d
from tqdm import tqdm

# --- I/O helpers ---

def is_macos_junk(p: Path) -> bool:
    """Weed out the macOS '._' AppleDouble files that otherwise wreck globbing."""
    return p.name.startswith("._")


def load_tile(path: Path) -> np.ndarray:
    """Load a 4-band float32 TIFF as (H, W, C), sorting out (C,H,W) vs (H,W,C) itself.

    HR files are stored (H,W,C); SR files are often (C,H,W). The heuristic: if the
    first axis is small (<=8) and the other two are large, it's channels-first.
    """
    img = tifffile.imread(str(path)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim == 3 and img.shape[0] <= 8 and img.shape[1] > 8 and img.shape[2] > 8:
        img = np.transpose(img, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    return img


def to_uint8_per_image(channel: np.ndarray) -> np.ndarray:
    """Per-image min-max into uint8 -- the baseline's existing convention."""
    lo, hi = float(channel.min()), float(channel.max())
    if hi - lo < 1e-12:
        return np.zeros_like(channel, dtype=np.uint8)
    return np.clip((channel - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def rgb_float01(img_hwc: np.ndarray) -> np.ndarray:
    """An (H,W,3) RGB image in [0,1], min-max per channel.

    Used for the network metrics (DISTS, CLIP) and SSIM-family, matching LPIPS prep.
    """
    rgb = img_hwc[:, :, :3]
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        lo, hi = float(ch.min()), float(ch.max())
        out[:, :, c] = 0.0 if hi - lo < 1e-12 else (ch - lo) / (hi - lo)
    return out


# --- structural metrics (no pretrained weights) ---

def psnr_uint8(a8: np.ndarray, b8: np.ndarray) -> float:
    mse = np.mean((a8.astype(np.float64) - b8.astype(np.float64)) ** 2)
    return 100.0 if mse < 1e-12 else 10.0 * np.log10((255.0 ** 2) / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def ssim_matlab(a8: np.ndarray, b8: np.ndarray) -> float:
    """SSIM, MATLAB-style: 11x11 Gaussian window, sigma=1.5, uint8 in."""
    a = a8.astype(np.float64)
    b = b8.astype(np.float64)
    K1, K2, L = 0.01, 0.03, 255.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    win = _gaussian_kernel(11, 1.5)
    mu_a = convolve2d(a, win, mode="valid")
    mu_b = convolve2d(b, win, mode="valid")
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sa = convolve2d(a * a, win, mode="valid") - mu_a2
    sb = convolve2d(b * b, win, mode="valid") - mu_b2
    sab = convolve2d(a * b, win, mode="valid") - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return float(ssim_map.mean())


def cw_ssim_band(c1: np.ndarray, c2: np.ndarray, K: float = 0.0) -> float:
    cw = c1 * np.conj(c2)
    num = 2.0 * np.abs(cw) + K
    den = np.abs(c1) ** 2 + np.abs(c2) ** 2 + K
    w = np.ones((7, 7), dtype=np.float64)
    w /= w.sum()
    num_p = convolve2d(num, w, mode="valid")
    den_p = convolve2d(den, w, mode="valid")
    return float(np.mean(num_p / (den_p + 1e-12)))


def cw_ssim(img1: np.ndarray, img2: np.ndarray, height: int = 5, order: int = 3) -> float:
    """Complex Wavelet SSIM on one 2D float channel. Higher = better."""
    import pyrtools as pt
    p1 = pt.pyramids.SteerablePyramidFreq(img1, height=height, order=order, is_complex=True)
    p2 = pt.pyramids.SteerablePyramidFreq(img2, height=height, order=order, is_complex=True)
    vals = [cw_ssim_band(p1.pyr_coeffs[k], p2.pyr_coeffs[k])
            for k in p1.pyr_coeffs if isinstance(k, tuple)]
    return float(np.mean(vals))


# --- perceptual metrics (lazy-loaded; want weights + torch) ---

class PerceptualBank:
    """Builds DISTS, MS-SSIM and CLIP on the chosen device, only once they're needed."""

    def __init__(self, device: str = "mps"):
        import torch
        self.torch = torch
        self.device = device
        self._dists = None
        self._clip = None
        self._clip_pre = None
        self._lpips = None

    def _to_tensor(self, rgb01: np.ndarray):
        # (H,W,3) [0,1] -> (1,3,H,W)
        t = self.torch.from_numpy(rgb01.transpose(2, 0, 1)).unsqueeze(0).float()
        return t.to(self.device)

    def dists(self, a01: np.ndarray, b01: np.ndarray) -> float:
        import piq
        if self._dists is None:
            self._dists = piq.DISTS().to(self.device).eval()
        with self.torch.no_grad():
            return float(self._dists(self._to_tensor(a01), self._to_tensor(b01)).item())

    def ms_ssim(self, a01: np.ndarray, b01: np.ndarray) -> float:
        import piq
        with self.torch.no_grad():
            return float(piq.multi_scale_ssim(
                self._to_tensor(a01), self._to_tensor(b01), data_range=1.0).item())

    def clip_cosine(self, a01: np.ndarray, b01: np.ndarray) -> float:
        """1 - cosine similarity of the two CLIP image embeddings. Lower = closer."""
        import open_clip
        if self._clip is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            self._clip = model.to(self.device).eval()
            self._clip_pre = preprocess
        from PIL import Image
        def embed(rgb01):
            arr = (rgb01 * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            x = self._clip_pre(pil).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                f = self._clip.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            return f
        fa, fb = embed(a01), embed(b01)
        cos = float((fa * fb).sum().item())
        return 1.0 - cos

    def lpips(self, a01: np.ndarray, b01: np.ndarray) -> float:
        """LPIPS (AlexNet). (H,W,3) in [0,1] in; the net wants [-1,1]. Lower = better."""
        import lpips as lpips_lib
        if self._lpips is None:
            self._lpips = lpips_lib.LPIPS(net="alex", verbose=False).to(self.device).eval()
        a = self._to_tensor(a01) * 2 - 1
        b = self._to_tensor(b01) * 2 - 1
        with self.torch.no_grad():
            return float(self._lpips(a, b).item())


# --- scoring one pair ---

def crop_to_match(hr: np.ndarray, sr: np.ndarray) -> np.ndarray:
    """Crop HR to SR's size, from the TOP-LEFT corner.

    The test set mixes 200x200 and 192x192 HR tiles, but SR is always 192x192.
    The alignment scan (check_crop_alignment.py) showed SR matches the TOP-LEFT
    192x192 of the bigger HR -- correlation peaks at offset (0,0) and drops off
    toward the center -- so the extra rows/cols come off the bottom and right.
    Center-cropping would slide things ~4px and quietly bias every metric. This is
    a same-resolution crop, no resampling, so nothing gets blurred by interpolation.
    """
    Hs, Ws = sr.shape[:2]
    Hh, Wh = hr.shape[:2]
    if (Hh, Wh) == (Hs, Ws):
        return hr
    if Hh >= Hs and Wh >= Ws:
        return hr[:Hs, :Ws, :]
    # HR smaller than SR somewhere -> shouldn't happen; let the shape check downstream complain
    return hr


CHANNEL_NAMES = ["Red", "Green", "Blue", "NIR"]


def score_pair(hr_path: Path, sr_path: Path, bank: PerceptualBank | None) -> dict:
    hr = load_tile(hr_path)
    sr = load_tile(sr_path)
    hr = crop_to_match(hr, sr)  # 200x200 HR down to 192x192, top-left, no resampling
    if hr.shape[:2] != sr.shape[:2]:
        raise ValueError(f"Shape mismatch {hr_path.name}: HR {hr.shape} vs SR {sr.shape}")

    row: dict[str, float] = {}

    # PSNR / SSIM / CW-SSIM per channel (CW-SSIM does NIR too)
    n_ch = min(hr.shape[2], sr.shape[2], 4)
    psnr_acc, ssim_acc = [], []
    for c in range(n_ch):
        a8 = to_uint8_per_image(hr[:, :, c])
        b8 = to_uint8_per_image(sr[:, :, c])
        p = psnr_uint8(a8, b8)
        s = ssim_matlab(a8, b8)
        cw = cw_ssim(a8.astype(np.float64) / 255.0, b8.astype(np.float64) / 255.0)
        name = CHANNEL_NAMES[c]
        row[f"PSNR_{name}"], row[f"SSIM_{name}"], row[f"CWSSIM_{name}"] = p, s, cw
        psnr_acc.append(p)
        ssim_acc.append(s)
    row["PSNR_Total"] = float(np.mean(psnr_acc))
    row["SSIM_Total"] = float(np.mean(ssim_acc))
    row["CWSSIM_Total"] = float(np.mean([row[f"CWSSIM_{CHANNEL_NAMES[c]}"] for c in range(n_ch)]))

    # the perceptual metrics, RGB only
    if bank is not None:
        hr01 = rgb_float01(hr)
        sr01 = rgb_float01(sr)
        row["MSSSIM_RGB"] = bank.ms_ssim(hr01, sr01)
        row["DISTS_RGB"] = bank.dists(hr01, sr01)
        row["CLIPcos_RGB"] = bank.clip_cosine(hr01, sr01)
        row["LPIPS_RGB"] = bank.lpips(hr01, sr01)

    return row


# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Subset of variant folder names; default = all.")
    ap.add_argument("--limit", type=int, default=None,
                    help="PoC: score only the first N tiles per variant.")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--no-perceptual", action="store_true",
                    help="Skip DISTS/CLIP/MS-SSIM (structural only; runs without weights).")
    args = ap.parse_args()

    import csv
    import pandas as pd

    hr_dir = Path(args.hr_dir).expanduser()
    results = Path(args.results).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_variant").mkdir(exist_ok=True)

    hr_index = {p.name: p for p in hr_dir.glob("*.tif") if not is_macos_junk(p)}
    print(f"HR tiles indexed: {len(hr_index)}")

    if args.variants:
        variant_dirs = [results / v for v in args.variants]
    else:
        variant_dirs = [d for d in results.iterdir() if d.is_dir()]

    bank = None if args.no_perceptual else PerceptualBank(device=args.device)

    summary_rows = []
    for vdir in variant_dirs:
        if not vdir.exists():
            print(f"  !! missing variant dir: {vdir}", file=sys.stderr)
            continue
        sr_files = sorted(p for p in vdir.glob("*.tif") if not is_macos_junk(p))
        if args.limit:
            sr_files = sr_files[:args.limit]
        print(f"\n[{vdir.name}] {len(sr_files)} tiles")

        rows, unreadable = [], []
        for sp in tqdm(sr_files, desc=vdir.name[:32], unit="tile", ncols=90):
            hp = hr_index.get(sp.name)
            if hp is None:
                unreadable.append((sp.name, "no HR match"))
                continue
            try:
                r = score_pair(hp, sp, bank)
                r["Filename"] = sp.name
                rows.append(r)
            except Exception as e:  # noqa: BLE001
                unreadable.append((sp.name, str(e)))

        if rows:
            df = pd.DataFrame(rows).set_index("Filename")
            df.to_csv(out / "per_variant" / f"{vdir.name}.csv")
            means = df.mean(numeric_only=True).to_dict()
            means["variant"] = vdir.name
            means["n_tiles"] = len(df)
            summary_rows.append(means)
        if unreadable:
            with open(out / "per_variant" / f"{vdir.name}_unreadable.csv", "w", newline="") as f:
                csv.writer(f).writerows([("filename", "reason"), *unreadable])
            print(f"   {len(unreadable)} unreadable/unmatched -> logged")

    if summary_rows:
        sdf = pd.DataFrame(summary_rows).set_index("variant")
        front = ["n_tiles", "PSNR_Total", "SSIM_Total", "CWSSIM_Total"]
        for opt in ["MSSSIM_RGB", "LPIPS_RGB", "DISTS_RGB", "CLIPcos_RGB"]:
            if opt in sdf.columns:
                front.append(opt)
        ordered = front + [c for c in sdf.columns if c not in front]
        sdf[ordered].to_csv(out / "all_variants_new_metrics_summary.csv")
        print(f"\nSummary written -> {out/'all_variants_new_metrics_summary.csv'}")
        print(sdf[front].round(4).to_string())


if __name__ == "__main__":
    main()
