"""
simulator.py -- forward model for the NPE (RQ1/RQ2) and for arm-2 training data (RQ3).

theta = (sigma_b, r, phi, sigma_n)          (simulated)
    sigma_b  Gaussian PSF blur, in PlanetScope pixels (3 m), applied on the HR grid
    r, phi   misregistration: magnitude (PlanetScope px) and direction (rad);
             dx = r cos(phi), dy = r sin(phi), applied as a cubic-spline sub-pixel shift
    sigma_n  Gaussian sensor noise, in reflectance units, added on the 10 m grid
The NPE infers (sigma_b, r, sigma_n): the metrics are symmetric, so the shift direction
is not identifiable from a fingerprint; phi is a nuisance (drawn at random for arm 2).

Forward model (the path Sentinel-2 data took to reach the PlanetScope grid):

    HR tile (H x H, reflectance)
      1. blur        gaussian_filter(sigma_b)                         per band, reflect
      2. shift       ndimage.shift((dy, dx), order=3)                 per band, reflect
      3. 10 m grid   resize to 60 x 60, bicubic, anti_aliasing=True   = the fake-LR recipe
      4. noise       + N(0, sigma_n^2), independent per pixel and band, on the 60 x 60 grid
      5. 3 m grid    resize back to H x H, bicubic                    (the data check: bicubic)

theta = 0 reproduces the fake-LR recipe exactly (192->60->192, or 200->60->200 on
200-px tiles, as in the fingerprints of the paper). Gain/offset is deliberately absent:
the P ruler min-max normalises every band, which removes any per-band affine change.

Fingerprint (P ruler, identical to the real Sentinel-2 / fake-LR pairs of the paper):
    both images -> per-band min-max to [0,1] -> PSNR / SSIM / CW-SSIM on all 4 bands
    (run_cluster_metrics.structural_row) + LPIPS / DISTS / MS-SSIM / CLIP-cos on the RGB
    and GBNIR passes, AVG = mean of the two (run_cluster_metrics.DualPassPerceptual).
"""
from __future__ import annotations

import importlib.util
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.transform import resize

HERE = Path(__file__).resolve().parent
LR_SIZE = 60

THETA_NAMES = ["sigma_b", "r", "phi", "sigma_n"]
INFER = ["sigma_b", "r", "sigma_n"]          # what the NPE estimates
# Prior: independent uniforms.  low, high
PRIOR = {
    "sigma_b": (0.0, 4.0),        # PlanetScope px; catalogue 0-3, S2 nearest defocus 3-4
    "r":       (0.0, 10 / 3),     # up to 1 Sentinel-2 pixel
    "phi":     (0.0, 2 * np.pi),  # direction, nuisance
    "sigma_n": (0.0, 0.02),       # reflectance
}
METRICS = ["PSNR", "SSIM", "CWSSIM", "LPIPS_AVG", "DISTS_AVG", "MSSSIM_AVG", "CLIPcos_AVG"]
PASS_COLS = [f"{m}_{p}" for m in ("LPIPS", "DISTS", "MSSSIM", "CLIPcos") for p in ("RGB", "GBNIR")]


# ----------------------------------------------------------------------------- prior
def sample_prior(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    lo = np.array([PRIOR[k][0] for k in THETA_NAMES])
    hi = np.array([PRIOR[k][1] for k in THETA_NAMES])
    return lo + (hi - lo) * rng.random((n, len(THETA_NAMES)))


def tile_seed(stem: str, draw: int) -> int:
    """Deterministic seed per (tile, draw): re-running gives identical records."""
    return (zlib.crc32(stem.encode()) * 1000 + draw) % (2 ** 32)


# ----------------------------------------------------------------------------- forward model
def _bicubic(a: np.ndarray, h: int, w: int, anti_aliasing: bool) -> np.ndarray:
    return resize(a, (h, w, a.shape[2]), order=3, mode="reflect",
                  anti_aliasing=anti_aliasing, preserve_range=True).astype(np.float32)


def degrade(hr: np.ndarray, theta, rng: np.random.Generator) -> np.ndarray:
    """hr: (H, W, 4) float reflectance.  Returns the degraded tile on the same grid."""
    sigma_b, r, phi, sigma_n = (float(t) for t in theta)
    dx, dy = r * np.cos(phi), r * np.sin(phi)
    H, W, C = hr.shape
    x = hr.astype(np.float32)
    if sigma_b > 0:
        x = np.stack([ndimage.gaussian_filter(x[:, :, c], sigma_b, mode="reflect")
                      for c in range(C)], axis=2)
    if r != 0:
        x = np.stack([ndimage.shift(x[:, :, c], (dy, dx), order=3, mode="reflect")
                      for c in range(C)], axis=2)
    lr = _bicubic(x, LR_SIZE, LR_SIZE, anti_aliasing=True)
    if sigma_n > 0:
        lr = lr + rng.normal(0.0, sigma_n, lr.shape).astype(np.float32)
    return _bicubic(lr, H, W, anti_aliasing=False)


# ----------------------------------------------------------------------------- fingerprint
def load_paper_modules():
    """The paper's metric code, copied unchanged into this folder."""
    def load(fname, name):
        spec = importlib.util.spec_from_file_location(name, HERE / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    A = load("evaluate_new_metrics.py", "evaluate_new_metrics")
    R = load("run_cluster_metrics.py", "run_cluster_metrics")
    return A, R


def unit(img: np.ndarray) -> np.ndarray:
    """Per-band min-max to [0,1] -- run_cluster_metrics.V_tile_to_unit_safe, minus the file read."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = float(ch.min()), float(ch.max())
        out[:, :, c] = 0.0 if hi - lo < 1e-12 else (ch - lo) / (hi - lo)
    return out


class Fingerprinter:
    def __init__(self, device: str = "cuda", perceptual: bool = True):
        self.A, self.R = load_paper_modules()
        self.bank = None
        if perceptual:
            self.bank = self.R.DualPassPerceptual(self.A, device=device)
            self.bank._PASSES = {"RGB": [0, 1, 2], "GBNIR": [1, 2, 3]}   # RRR control not needed

    def __call__(self, ref: np.ndarray, deg: np.ndarray) -> dict:
        r, d = unit(ref), unit(deg)
        row = dict(self.R.structural_row(r, d, self.A))
        if self.bank is not None:
            p = self.bank.score_pair_dual(r, d)
            row.update({k: p[k] for k in PASS_COLS + [m for m in METRICS if m.endswith("_AVG")]})
        return row
