#!/usr/bin/env python3
"""
build_arm2.py -- RQ3 arm 2 input tiles: PlanetScope degraded "like Sentinel-2", one recipe per tile,
with theta drawn from the NPE posterior of REAL Sentinel-2 pairs.

    python build_arm2.py --samples real/post_s2_samples.npz --source pool   [--radiometry pair]
    python build_arm2.py --samples post_train_samples.npz   --source matched

--source pool     each tile gets theta from the posterior of a RANDOM real Sentinel-2 tile in the npz
                  (the population of real degradations; works with the paper's 3,000 real pairs)
--source matched  each tile gets theta from the posterior of ITS OWN real Sentinel-2 pair
                  (needs posteriors for the RQ3 tiles: simulate.py --real + npe.py apply)
In both cases one posterior sample is drawn at random (not the mean), so arm 2 inherits the
posterior's uncertainty; the shift direction phi is drawn uniformly (not identifiable).

--radiometry none   (default) only blur / shift / noise, exactly the NPE's forward model
--radiometry pair   additionally map each band with the gain/offset measured on the SAME real pair
                    the theta came from: least squares S2 = a * fakeLR(PS) + b on the raw reflectances.
                    The fingerprint (P ruler) cannot see gain/offset, but the raw pair can.

Tiles: train + val + test stems of the RQ3 splits, PlanetScope cropped to 192 x 192 first (as for
arms 1 and 3). theta = 0 reproduces rq3_fake_lr exactly.  Output: $RQ3_DATA/rq3_s2like/<stem>.tif
(float32, H,W,C, the same TIFF layout as the other inputs) + rq3_s2like_recipes.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator as S  # noqa: E402

SIZE = 192


def to_hwc(a):
    a = np.asarray(a, dtype=np.float32)
    return a if a.shape[-1] == 4 else np.moveaxis(a, 0, -1)


def gain_offset(hr_raw, s2_raw):
    """Per band least squares s2 = a * fakeLR(hr) + b, on the full native tiles of one real pair."""
    fl = S.degrade(hr_raw, (0, 0, 0, 0), np.random.default_rng(0))
    ab = []
    for c in range(4):
        x, y = fl[..., c].ravel(), s2_raw[..., c].ravel()
        A = np.stack([x, np.ones_like(x)], 1)
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        ab.append((float(a), float(b)))
    return ab


def save(path, arr):
    tifffile.imwrite(path, np.ascontiguousarray(arr, dtype=np.float32), photometric="rgb",
                     planarconfig="contig", extrasamples=(2,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="npz from npe.py apply --samples-out")
    ap.add_argument("--source", choices=["pool", "matched"], default="pool")
    ap.add_argument("--radiometry", choices=["none", "pair"], default="none")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--region", default="held_out",
                    help="pool mode: only use real tiles of this region (column in real/sentinel.csv); '' = all")
    ap.add_argument("--real-table", default=None, help="real/sentinel.csv (for --region filtering)")
    ap.add_argument("--out", default=None, help="default $RQ3_DATA/rq3_s2like")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    D, T = Path(os.environ["RQ3_DATA"]), Path(os.environ["RQ3_TEST"])
    splits_dir = Path(os.environ.get("RQ3_SPLITS", Path(__file__).resolve().parent / "splits"))
    out = Path(a.out or D / "rq3_s2like"); out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.samples, allow_pickle=False)
    tiles, samples, names = list(z["tiles"]), z["samples"], list(z["names"])
    assert names == S.INFER, f"posterior is over {names}, simulator infers {S.INFER}"
    idx_of = {t: i for i, t in enumerate(tiles)}
    pool = list(range(len(tiles)))
    if a.source == "pool" and a.region and a.real_table:
        import pandas as pd
        reg = pd.read_csv(a.real_table).set_index("tile")["region"].to_dict()
        pool = [i for i, t in enumerate(tiles) if reg.get(t) == a.region]
    if not pool:
        raise SystemExit("no source tiles available")
    print(f"posterior samples: {len(tiles)} real tiles x {samples.shape[1]}; source={a.source} "
          f"(pool of {len(pool)}); radiometry={a.radiometry}")

    jobs = []
    for sp in a.splits:
        root = T if sp == "test" else D
        for r in csv.DictReader(open(splits_dir / f"{sp}.csv")):
            jobs.append((sp, r["stem"], root))
    if a.limit:
        jobs = jobs[: a.limit]

    rec_path = out.parent / f"{out.name}_recipes.csv"
    cols = ["split", "stem", "source_tile", "sample", "sigma_b", "r", "phi", "sigma_n"] + \
           [f"{k}{c}" for c in range(4) for k in ("gain", "offset")]
    fh = open(rec_path, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    ab_cache = {}
    for n, (sp, stem, root) in enumerate(jobs, 1):
        rng = np.random.default_rng(zlib.crc32(f"arm2|{stem}".encode()))
        if a.source == "matched":
            if stem not in idx_of:
                raise SystemExit(f"--source matched: no posterior for {stem}")
            src = idx_of[stem]
        else:
            src = pool[rng.integers(len(pool))]
        k = int(rng.integers(samples.shape[1]))
        th = dict(zip(S.INFER, map(float, samples[src, k])))
        theta = (th["sigma_b"], th["r"], float(rng.uniform(0, 2 * np.pi)), th["sigma_n"])
        hr = to_hwc(tifffile.imread(root / "hr_200" / f"{stem}.tif"))[:SIZE, :SIZE, :4]
        deg = S.degrade(hr, theta, rng)
        ab = [(1.0, 0.0)] * 4
        if a.radiometry == "pair":
            st = tiles[src]
            if st not in ab_cache:
                sroot = T if (T / "hr_200" / f"{st}.tif").exists() else D
                ab_cache[st] = gain_offset(to_hwc(tifffile.imread(sroot / "hr_200" / f"{st}.tif"))[..., :4],
                                           to_hwc(tifffile.imread(sroot / "sr_60_200" / f"{st}.tif"))[..., :4])
            ab = ab_cache[st]
            deg = np.stack([ab[c][0] * deg[..., c] + ab[c][1] for c in range(4)], axis=2)
        save(out / f"{stem}.tif", deg)
        w.writerow({"split": sp, "stem": stem, "source_tile": tiles[src], "sample": k,
                    "sigma_b": theta[0], "r": theta[1], "phi": theta[2], "sigma_n": theta[3],
                    **{f"gain{c}": ab[c][0] for c in range(4)}, **{f"offset{c}": ab[c][1] for c in range(4)}})
        if n % 500 == 0 or n == len(jobs):
            fh.flush(); print(f"  {n}/{len(jobs)}", flush=True)
    fh.close()
    print(f"-> {out}  ({len(jobs)} tiles)  recipes: {rec_path}")


if __name__ == "__main__":
    main()
