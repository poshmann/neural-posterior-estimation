#!/usr/bin/env python3
"""
simulate.py -- make the (theta, fingerprint) records for the NPE.

    python simulate.py --split train --draws 10 --device cuda --out sims/train.csv
    python simulate.py --split train --draws 10 --shard 0 --nshards 4 ...   # parallel pieces
    python simulate.py --split train --n-tiles 20 --draws 5 --out sims/timing.csv   # timing test

* tiles: the RQ3 split lists (splits/<split>.csv), HR from $RQ3_DATA/hr_200, at native size
  (192 or 200 px) -- the same tiles and sizes the paper's real fingerprints used
* every (tile, draw) has a fixed seed: theta and the noise are reproducible, re-runs resume
* one row per record: stem, H, draw, seed, theta..., PSNR, SSIM, CWSSIM, *_RGB, *_GBNIR, *_AVG, seconds
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--draws", type=int, default=10, help="theta draws per tile")
    ap.add_argument("--n-tiles", type=int, default=0, help="only the first N tiles (0 = all)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-perceptual", action="store_true", help="structural metrics only (testing)")
    ap.add_argument("--baseline", action="store_true",
                    help="one theta=0 (fake-LR) record per tile, columns b_<metric>: scene context for the NPE")
    ap.add_argument("--real", action="store_true",
                    help="fingerprint the REAL Sentinel-2 pair (sr_60_200 vs hr_200) of each tile, no theta")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    data = Path(os.environ["RQ3_TEST" if a.split == "test" else "RQ3_DATA"]) / "hr_200"
    splits = Path(os.environ.get("RQ3_SPLITS", Path(__file__).resolve().parent / "splits"))
    stems = [r["stem"] for r in csv.DictReader(open(splits / f"{a.split}.csv"))]
    if a.n_tiles:
        stems = stems[:a.n_tiles]
    stems = stems[a.shard::a.nshards]

    out = Path(a.out)
    if a.nshards > 1:
        out = out.with_name(f"{out.stem}_shard{a.shard:02d}of{a.nshards:02d}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size:
        done = {(r["stem"], int(r["draw"])) for r in csv.DictReader(open(out))}

    fp = S.Fingerprinter(device=a.device, perceptual=not a.no_perceptual)
    metric_cols = ["PSNR", "SSIM", "CWSSIM"] + ([] if a.no_perceptual else
                   S.PASS_COLS + [m for m in S.METRICS if m.endswith("_AVG")])
    cols = ["stem", "H", "draw", "seed"] + S.THETA_NAMES + \
        ([f"b_{k}" for k in metric_cols] if a.baseline else metric_cols) + ["seconds"]
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if not done:
        w.writeheader()

    if a.baseline or a.real:
        a.draws = 1
    todo = [(s, d) for s in stems for d in range(a.draws) if (s, d) not in done]
    print(f"{a.split}: {len(stems)} tiles x {a.draws} draws -> {len(todo)} records to do "
          f"({len(done)} already in {out})", flush=True)
    t_start = time.time()
    hr_cache = (None, None)
    for i, (stem, d) in enumerate(todo, 1):
        t0 = time.time()
        if hr_cache[0] != stem:
            hr_cache = (stem, fp.A.load_tile(data / f"{stem}.tif"))   # paper's loader, native size
        hr = hr_cache[1]
        seed = S.tile_seed(stem, d)
        rng = np.random.default_rng(seed)
        theta = np.zeros(len(S.THETA_NAMES)) if a.baseline else S.sample_prior(rng)[0]
        if a.real:
            theta = np.full(len(S.THETA_NAMES), np.nan)
            deg = fp.A.load_tile(data.parent / "sr_60_200" / f"{stem}.tif")     # real Sentinel-2
            if deg.shape != hr.shape:
                raise SystemExit(f"{stem}: S2 {deg.shape} vs PS {hr.shape}")
        else:
            deg = S.degrade(hr, theta, rng)
        row = fp(hr, deg)
        w.writerow({"stem": stem, "H": hr.shape[0], "draw": d, "seed": seed,
                    **dict(zip(S.THETA_NAMES, map(float, theta))),
                    **{(("b_" + k) if a.baseline else k): float(row[k]) for k in metric_cols},
                    "seconds": round(time.time() - t0, 3)})
        if i % 50 == 0 or i == len(todo):
            fh.flush()
            el = time.time() - t_start
            print(f"  {i}/{len(todo)}  {el / i:.2f} s/record  ~{el / i * (len(todo) - i) / 60:.0f} min left",
                  flush=True)
    fh.close()


if __name__ == "__main__":
    main()
