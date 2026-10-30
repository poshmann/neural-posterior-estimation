#!/usr/bin/env python3
"""
radiometry_fits.py -- how well does a per-band gain/offset describe real Sentinel-2 vs PlanetScope?
Diagnostic for build_arm2.py --radiometry (decide: none / global / per-pair with a quality gate).

    python radiometry_fits.py --real-table $RQ3_RUNS/npe_v3/sentinel.csv --region held_out --out $RQ3_RUNS/radiometry
    python radiometry_fits.py --stems splits/train.csv --out $RQ3_RUNS/radiometry_aoi_train   # RQ3 train pairs (all AOI_*)

For every real pair (PlanetScope hr_200 + Sentinel-2 sr_60_200, native tile size, same lookup as
build_arm2.py) and every band c, with x = fakeLR(PS) (theta = 0, exactly what build_arm2 fits) and y = S2:
  corr        Pearson r between x and y  (how "line-like" vs "blob-like" the pair is)
  ols_*       least squares y = a x + b   (= what build_arm2 --radiometry pair uses now)
  rma_*       reduced major axis: a = sign(r) sd_y/sd_x, b = mean_y - a mean_x   (no regression dilution)
  mean_*, sd_*  per-band mean / std of x and y
Pixels used: finite in both images.

Outputs (in --out):
  radiometry_per_tile.csv   one row per tile x band
  radiometry_summary.txt    per band: corr quantiles, share of blob-like pairs, gain/offset quantiles,
                            share of OLS gains outside [0.5, 1.5], share giving negative values when
                            transferred to other tiles (pool mode), and global candidates
  radiometry.png            per band: histogram of corr, histogram of OLS vs RMA gain, tile means S2 vs fakeLR
  radiometry_global.json    per band global candidates; "gain"/"offset" = A2 (median RMA of line-like pairs),
                            the input for build_arm2.py --radiometry global; plus "q_fakelr"/"q_s2": pooled pixel
                            quantiles (1001 levels) of fakeLR and S2 per band, the input for --radiometry quantile
                            (global quantile mapping = CDF matching; 29 Oct 2026: the AOI relation is close to
                            multiplicative, a global straight line gave 6 % negative pixels vs 0.06 % in real S2)
Note (29 Oct 2026): the data has two tile families. AOI_* pairs: S2 darker than PlanetScope; tile_* (scene)
pairs: S2 brightness already matched. RQ3 train/val are all AOI_*, so arm 2 uses the fit on splits/train.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator as S  # noqa: E402

BANDS = ["b0", "b1", "b2", "b3"]   # file order; band names not confirmed (R,G,B vs B,G,R open), b3 = NIR


def to_hwc(a):
    a = np.asarray(a, dtype=np.float32)
    return a if a.shape[-1] == 4 else np.moveaxis(a, 0, -1)


def band_stats(x, y):
    """x, y: 1-D arrays of one band. Returns dict of fit statistics (NaN if degenerate)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m].astype(np.float64), y[m].astype(np.float64)
    out = dict(n=int(m.sum()), mean_x=np.nan, mean_y=np.nan, sd_x=np.nan, sd_y=np.nan, corr=np.nan,
               ols_gain=np.nan, ols_off=np.nan, rma_gain=np.nan, rma_off=np.nan)
    if x.size < 10:
        return out
    mx, my, sx, sy = x.mean(), y.mean(), x.std(), y.std()
    out.update(mean_x=mx, mean_y=my, sd_x=sx, sd_y=sy)
    if sx <= 1e-8 or sy <= 1e-8:
        return out
    r = float(np.mean((x - mx) * (y - my)) / (sx * sy))
    a_ols = r * sy / sx
    a_rma = np.sign(r) * sy / sx if r != 0 else np.nan
    out.update(corr=r, ols_gain=a_ols, ols_off=my - a_ols * mx, rma_gain=a_rma, rma_off=my - a_rma * mx)
    return out


def q(s, ps=(5, 25, 50, 75, 95)):
    s = pd.Series(s).dropna()
    return "  ".join(f"p{p} {np.percentile(s, p):+.3f}" for p in ps) if len(s) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-table", default=None, help="real/sentinel.csv (tile, region, ...)")
    ap.add_argument("--stems", default=None, help="instead of --real-table: a split csv with a 'stem' column "
                                                  "(e.g. splits/train.csv); tiles read from $RQ3_DATA")
    ap.add_argument("--region", default="held_out", help="only tiles of this region; '' = all")
    ap.add_argument("--good-corr", type=float, default=0.8, help="corr threshold for a 'line-like' pair")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D, T = Path(os.environ["RQ3_DATA"]), Path(os.environ["RQ3_TEST"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.stems:
        tiles = [r["stem"] for r in csv.DictReader(open(a.stems))]
        a.region = ""
    elif a.real_table:
        tab = pd.read_csv(a.real_table)
        if a.region:
            tab = tab[tab["region"] == a.region]
        tiles = tab["tile"].tolist()
    else:
        raise SystemExit("give --real-table or --stems")
    if a.limit:
        tiles = tiles[: a.limit]
    print(f"{len(tiles)} pairs ({'stems ' + a.stems if a.stems else 'region=' + (a.region or 'all')})", flush=True)

    rows, missing = [], 0
    QN, PX = 1001, 600                       # quantile levels; random pixels per tile and band for the pool
    pool_x, pool_y = [[] for _ in range(4)], [[] for _ in range(4)]
    prng = np.random.default_rng(0)
    for n, st in enumerate(tiles, 1):
        root = T if (T / "hr_200" / f"{st}.tif").exists() else D
        try:
            hr = to_hwc(tifffile.imread(root / "hr_200" / f"{st}.tif"))[..., :4]
            s2 = to_hwc(tifffile.imread(root / "sr_60_200" / f"{st}.tif"))[..., :4]
        except FileNotFoundError:
            missing += 1
            continue
        fl = S.degrade(hr, (0, 0, 0, 0), np.random.default_rng(0))
        for c in range(4):
            xs, ys = fl[..., c].ravel(), s2[..., c].ravel()
            rows.append(dict(tile=st, band=BANDS[c], H=hr.shape[0], **band_stats(xs, ys)))
            ok = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
            if ok.size:
                pick = prng.choice(ok, size=min(PX, ok.size), replace=False)
                pool_x[c].append(xs[pick].astype(np.float32)); pool_y[c].append(ys[pick].astype(np.float32))
        if n % 250 == 0 or n == len(tiles):
            print(f"  {n}/{len(tiles)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "radiometry_per_tile.csv", index=False)

    L = [f"radiometry_fits.py  pairs used: {df.tile.nunique()}  missing files: {missing}  region: {a.region or 'all'}",
         "x = fakeLR(PlanetScope) (theta=0), y = Sentinel-2, per band, native tile; b3 = NIR",
         f"'line-like' pair = corr >= {a.good_corr}", ""]
    rng = np.random.default_rng(0)
    glob_ = {"source": a.stems or f"{a.real_table} region={a.region or 'all'}", "pairs": int(df.tile.nunique()),
             "good_corr": a.good_corr, "estimator": "A2 median RMA of line-like pairs", "bands": BANDS,
             "gain": [], "offset": [], "A1_ols_gain": [], "A1_ols_offset": [], "A3_means_gain": [], "A3_means_offset": [],
             "quantile_levels": QN, "pixels_per_tile_band": PX, "q_fakelr": [], "q_s2": []}
    levels = np.linspace(0, 100, QN)
    for c in range(4):
        x = np.concatenate(pool_x[c]); y = np.concatenate(pool_y[c])
        glob_["q_fakelr"].append([float(v) for v in np.percentile(x, levels)])
        glob_["q_s2"].append([float(v) for v in np.percentile(y, levels)])
    for b in BANDS:
        d = df[df.band == b].dropna(subset=["corr"])
        good = d[d["corr"] >= a.good_corr]
        L += [f"== {b}  (n {len(d)})",
              f"  corr          {q(d['corr'])}",
              f"  share corr<0 {np.mean(d['corr'] < 0):.3f}   <0.5 {np.mean(d['corr'] < 0.5):.3f}   "
              f"<{a.good_corr} {np.mean(d['corr'] < a.good_corr):.3f}",
              f"  OLS gain      {q(d.ols_gain)}",
              f"  OLS offset    {q(d.ols_off)}",
              f"  share OLS gain outside [0.5, 1.5]: {np.mean((d.ols_gain < 0.5) | (d.ols_gain > 1.5)):.3f}",
              f"  RMA gain      {q(d.rma_gain)}",
              f"  RMA offset    {q(d.rma_off)}",
              f"  sd ratio y/x  {q(d.sd_y / d.sd_x)}   mean diff y-x {q(d.mean_y - d.mean_x)}"]
        # pool-mode transfer: fit of a random source pair applied to a random OTHER tile's mean
        if len(d) > 1:
            src = d.sample(n=min(5000, 20 * len(d)), replace=True, random_state=0)
            tgt = d.mean_x.to_numpy()[rng.integers(len(d), size=len(src))]
            neg = np.mean(src.ols_gain.to_numpy() * tgt + src.ols_off.to_numpy() < 0)
            L.append(f"  pool transfer, OLS of random pair on random tile mean: share mean < 0 = {neg:.3f}")
        # global candidates
        gm = np.polyfit(d.mean_x, d.mean_y, 1) if len(d) > 2 else (np.nan, np.nan)
        L += [f"  global A1  median OLS of line-like pairs (n {len(good)}): gain {good.ols_gain.median():.3f}  "
              f"offset {good.ols_off.median():+.4f}",
              f"  global A2  median RMA of line-like pairs:               gain {good.rma_gain.median():.3f}  "
              f"offset {good.rma_off.median():+.4f}",
              f"  global A3  line through tile MEANS (all pairs):        gain {gm[0]:.3f}  offset {gm[1]:+.4f}", ""]
        glob_["gain"].append(float(good.rma_gain.median())); glob_["offset"].append(float(good.rma_off.median()))
        glob_["A1_ols_gain"].append(float(good.ols_gain.median())); glob_["A1_ols_offset"].append(float(good.ols_off.median()))
        glob_["A3_means_gain"].append(float(gm[0])); glob_["A3_means_offset"].append(float(gm[1]))
        qx, qy = np.array(glob_["q_fakelr"][BANDS.index(b)]), np.array(glob_["q_s2"][BANDS.index(b)])
        L.insert(len(L) - 1, "  quantile map S2/fakeLR at p1 p25 p50 p75 p99: " + "  ".join(
            f"{qy[i] / qx[i]:.2f}" if qx[i] > 1e-4 else "n/a" for i in (10, 250, 500, 750, 990)))
    (out / "radiometry_global.json").write_text(json.dumps(glob_, indent=2))
    txt = "\n".join(L)
    (out / "radiometry_summary.txt").write_text(txt)
    print(txt)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 4, figsize=(16, 11))
        for j, b in enumerate(BANDS):
            d = df[df.band == b].dropna(subset=["corr"])
            ax[0, j].hist(d["corr"], bins=60, range=(-1, 1), color="0.4")
            ax[0, j].axvline(a.good_corr, color="C3", ls="--"); ax[0, j].set_title(f"{b}: corr(fakeLR, S2)")
            ax[1, j].hist(d.ols_gain.clip(-1, 3), bins=80, alpha=.6, label="OLS")
            ax[1, j].hist(d.rma_gain.clip(-1, 3), bins=80, alpha=.6, label="RMA")
            ax[1, j].axvline(1, color="k", lw=.8); ax[1, j].set_title(f"{b}: gain (clipped to [-1, 3])"); ax[1, j].legend()
            ax[2, j].scatter(d.mean_x, d.mean_y, s=3, c=d["corr"], cmap="viridis", vmin=0, vmax=1)
            lim = [0, max(d.mean_x.max(), d.mean_y.max())]
            ax[2, j].plot(lim, lim, "k", lw=.8); ax[2, j].set_xlabel("tile mean fakeLR"); ax[2, j].set_ylabel("tile mean S2")
        fig.tight_layout(); fig.savefig(out / "radiometry.png", dpi=110)
        print(f"-> {out}/radiometry.png")
    except Exception as e:  # plotting is optional
        print("no plot:", e)
    print(f"-> {out}/radiometry_per_tile.csv, radiometry_summary.txt, radiometry_global.json")


if __name__ == "__main__":
    main()
