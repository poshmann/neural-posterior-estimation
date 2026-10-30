#!/usr/bin/env python3
"""
shift_check.py -- independent check of the NPE shift r: measure the S2-vs-PlanetScope offset directly.

The NPE infers |shift| r from the 7-metric fingerprint only. Here the same offset is measured from the
pixels with phase correlation between fakeLR(PS) (theta = 0, as in build_arm2/radiometry_fits) and S2
(sr_60_200, same grid). If the NPE r tracks the measured shift tile by tile, r is real misregistration;
if not (e.g. measured shift small but r at the prior edge), r absorbs something else (scene change,
radiometry, texture) -> misspecification.

    python shift_check.py --real-table $RQ3_RUNS/npe_v3/sentinel.csv --region held_out \
        --post $RQ3_RUNS/npe_v4/post_sentinel.csv --out $RQ3_RUNS/shift_check

Method per tile: each band standardised, Hann window, normalised cross-power spectrum summed over the
4 bands, integer peak + parabolic sub-pixel refinement. Positive dx/dy = S2 content lies at +x/+y
relative to PS (same sign as simulator.degrade). 'peak' = height of the correlation peak (0..1);
low peak = no clear common structure, the shift is then unreliable.

Outputs: shift_check.csv (tile, dx, dy, shift, peak, [+ NPE r_q05/q50/q95]), shift_check.txt, shift_check.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator as S  # noqa: E402


def to_hwc(a):
    a = np.asarray(a, dtype=np.float32)
    return a if a.shape[-1] in (3, 4, 5) and a.shape[0] > 8 else np.moveaxis(a, 0, -1)


def _parabola(cm, c0, cp):
    d = cm - 2 * c0 + cp
    return 0.0 if abs(d) < 1e-12 else 0.5 * (cm - cp) / d


def phase_shift(ref, mov, max_shift=12, fmax=0.15):
    """ref, mov: (H, W, C). Returns (dx, dy, peak): mov(x) ~ ref(x - (dx, dy))."""
    H, W, C = ref.shape
    win = np.outer(np.hanning(H), np.hanning(W))
    # only frequencies Sentinel-2 actually carries (below its Nyquist, 1/6 cycle per PS px): above that
    # the S2 / fakeLR images hold only resampling structure on the 3-px S2 grid, which snaps the peak
    # to multiples of 3 PS px
    fy, fx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")
    keep = np.hypot(fx, fy) < fmax
    R = np.zeros((H, W), complex)
    for c in range(C):
        a, b = ref[..., c].astype(np.float64), mov[..., c].astype(np.float64)
        m = np.isfinite(a) & np.isfinite(b)
        if m.mean() < 0.9:
            continue
        a = np.where(m, a, np.nanmean(a)); b = np.where(m, b, np.nanmean(b))
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        a = (a - a.mean()) / a.std() * win
        b = (b - b.mean()) / b.std() * win
        Fa, Fb = np.fft.fft2(a), np.fft.fft2(b)
        X = Fb * np.conj(Fa)
        R += np.where(keep, X / (np.abs(X) + 1e-12), 0)
    if not np.any(R):
        return np.nan, np.nan, np.nan
    corr = np.real(np.fft.ifft2(R)) * H * W / (C * keep.sum())     # 1 = identical up to a shift
    corr = np.fft.fftshift(corr)
    cy, cx = H // 2, W // 2
    sub = corr[cy - max_shift: cy + max_shift + 1, cx - max_shift: cx + max_shift + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    py, px = iy + cy - max_shift, ix + cx - max_shift
    fy = _parabola(corr[py - 1, px], corr[py, px], corr[(py + 1) % H, px])
    fx = _parabola(corr[py, px - 1], corr[py, px], corr[py, (px + 1) % W])
    return float(px - cx + fx), float(py - cy + fy), float(corr[py, px])


def selftest():
    rng = np.random.default_rng(0)
    from scipy import ndimage
    hr = ndimage.gaussian_filter(rng.random((192, 192, 4)), (2, 2, 0)).astype(np.float32)
    ref = S.degrade(hr, (0, 0, 0, 0), np.random.default_rng(0))
    errs = []
    for _ in range(20):
        sb, r, phi, sn = rng.uniform(0, 3), rng.uniform(0, 20 / 3), rng.uniform(0, 2 * np.pi), 0.002
        mov = S.degrade(hr, (sb, r, phi, sn), np.random.default_rng(1))
        dx, dy, pk = phase_shift(ref, mov)
        errs.append(np.hypot(dx - r * np.cos(phi), dy - r * np.sin(phi)))
        print(f"  true r {r:5.2f} phi {phi:4.2f} sb {sb:4.2f} -> measured {np.hypot(dx, dy):5.2f} "
              f"(dx {dx:+5.2f} vs {r*np.cos(phi):+5.2f}, dy {dy:+5.2f} vs {r*np.sin(phi):+5.2f}) peak {pk:.2f}")
    print(f"selftest: median vector error {np.median(errs):.3f} PS px, max {np.max(errs):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-table", help="real/sentinel.csv (tile, region, group)")
    ap.add_argument("--stems", help="alternative: csv with a stem/tile column (e.g. splits/test.csv)")
    ap.add_argument("--region", default="held_out")
    ap.add_argument("--post", default=None, help="npe.py apply output (tile, r_q05, r_q50, r_q95) to compare with")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import tifffile

    D, T = Path(os.environ["RQ3_DATA"]), Path(os.environ["RQ3_TEST"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.real_table:
        tab = pd.read_csv(a.real_table)
        if a.region and "region" in tab:
            tab = tab[tab["region"] == a.region]
    else:
        tab = pd.read_csv(a.stems)
        tab = tab.rename(columns={"stem": "tile"}) if "stem" in tab else tab
    tiles = tab["tile"].astype(str).str.replace(".tif", "", regex=False).tolist()
    if a.limit:
        tiles = tiles[: a.limit]
    print(f"{len(tiles)} pairs", flush=True)
    rows, missing = [], 0
    for n, st in enumerate(tiles, 1):
        root = T if (T / "hr_200" / f"{st}.tif").exists() else D
        try:
            hr = to_hwc(tifffile.imread(root / "hr_200" / f"{st}.tif"))[..., :4]
            s2 = to_hwc(tifffile.imread(root / "sr_60_200" / f"{st}.tif"))[..., :4]
        except FileNotFoundError:
            missing += 1
            continue
        fl = S.degrade(hr, (0, 0, 0, 0), np.random.default_rng(0))
        dx, dy, pk = phase_shift(fl, s2)
        rows.append(dict(tile=st, H=hr.shape[0], dx=dx, dy=dy, shift=np.hypot(dx, dy), peak=pk))
        if n % 250 == 0 or n == len(tiles):
            print(f"  {n}/{len(tiles)}", flush=True)
    df = pd.DataFrame(rows)
    meta = [c for c in ("tile", "group", "region") if c in tab]
    df = df.merge(tab[meta].assign(tile=tab["tile"].astype(str)), on="tile", how="left")
    df["family"] = np.where(df.tile.str.startswith("AOI"), "AOI", "tile_")
    if a.post:
        p = pd.read_csv(a.post)[["tile", "r_q05", "r_q50", "r_q95"]]
        df = df.merge(p.assign(tile=p.tile.astype(str)), on="tile", how="left")
    df.to_csv(out / "shift_check.csv", index=False)

    L = [f"shift_check.py  pairs {len(df)}  missing files {missing}",
         "measured shift = phase correlation fakeLR(PS) vs S2 [PS px]; peak = correlation peak height", ""]
    for f, d in [("all", df)] + [(f, df[df.family == f]) for f in ("AOI", "tile_")]:
        if not len(d):
            continue
        qs = d["shift"].quantile([.1, .25, .5, .75, .9]).round(2).tolist()
        L.append(f"== {f} (n {len(d)})  measured shift p10/25/50/75/90 {qs}   peak median {d.peak.median():.2f}")
        if "r_q50" in d:
            ok = d.dropna(subset=["r_q50", "shift"])
            from scipy.stats import spearmanr
            rho = spearmanr(ok["shift"], ok.r_q50).correlation
            inside = np.mean((ok["shift"] >= ok.r_q05) & (ok["shift"] <= ok.r_q95))
            L.append(f"   NPE r_q50 median {ok.r_q50.median():.2f}   Spearman(measured, r_q50) {rho:+.2f}   "
                     f"measured inside NPE 90% interval {inside:.2f}   median(r_q50 - measured) "
                     f"{(ok.r_q50 - ok['shift']).median():+.2f}")
            e = ok[ok.r_q50 > 6.0]
            if len(e):
                L.append(f"   tiles with r_q50 > 6 (prior top 10%): {len(e)}, their measured shift median "
                         f"{e['shift'].median():.2f}, peak median {e.peak.median():.2f}")
    if "group" in df:
        g = df.groupby("group").agg(n=("tile", "size"), dx=("dx", "median"), dy=("dy", "median"),
                                    sd_dx=("dx", "std"), sd_dy=("dy", "std"), shift=("shift", "median"))
        g = g[g.n >= 10].sort_values("shift", ascending=False)
        L += ["", "groups with >= 10 tiles, largest median shift first (a common dx, dy with small sd within a",
              "group = one offset for the whole scene/AOI, i.e. georeferencing):",
              g.head(15).round(2).to_string()]
    txt = "\n".join(L)
    (out / "shift_check.txt").write_text(txt + "\n")
    print(txt)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].scatter(df.dx, df.dy, s=3, c=(df.family == "AOI"), cmap="coolwarm", alpha=.6)
        ax[0].set_xlabel("dx [PS px]"); ax[0].set_ylabel("dy [PS px]"); ax[0].set_aspect("equal")
        ax[0].set_title("measured offset S2 vs PS (red = AOI)")
        if "r_q50" in df:
            ax[1].scatter(df["shift"], df.r_q50, s=3, alpha=.5)
            ax[1].plot([0, 7], [0, 7], "k", lw=.8)
            ax[1].set_xlabel("measured shift [PS px]"); ax[1].set_ylabel("NPE r posterior median [PS px]")
        fig.tight_layout(); fig.savefig(out / "shift_check.png", dpi=120)
    except Exception as e:  # noqa: BLE001
        print("no plot:", e)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
