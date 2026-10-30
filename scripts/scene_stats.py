#!/usr/bin/env python3
"""
scene_stats.py -- per-tile scene statistics of the clean PlanetScope tile (reflectance, native size):
per-band standard deviation (hr_std0..3) and mean (hr_mean0..3).

Why: the P ruler min-max normalises every image, so a fingerprint cannot tell how strong ABSOLUTE noise
(sigma_n, reflectance) is relative to the scene. Without it the NPE over-estimates noise on very flat
tiles and under-estimates it on high-contrast tiles (found 29 Oct 2026). npe.py --scene uses log10(hr_std).

    source env.sh && python scene_stats.py --real real_paper/sentinel_per_tile.csv --out scene_stats.csv

Tiles: RQ3 train + val + test splits, plus every tile listed in --real (the paper's real pairs).
Looks for <stem>.tif in $RQ3_DATA/hr_200, then $RQ3_TEST/hr_200. Resumable.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="*", default=[], help="csv(s) with a 'tile' column (e.g. sentinel_per_tile.csv)")
    ap.add_argument("--splits", default=os.environ.get("RQ3_SPLITS", str(HERE / "splits")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    roots = [Path(os.environ["RQ3_DATA"]) / "hr_200", Path(os.environ["RQ3_TEST"]) / "hr_200"]

    stems = [r["stem"] for sp in ("train", "val", "test")
             for r in csv.DictReader(open(Path(a.splits) / f"{sp}.csv"))]
    for f in a.real:
        stems += [str(t).replace(".tif", "") for t in pd.read_csv(f)["tile"]]
    stems = list(dict.fromkeys(stems))
    out = Path(a.out)
    done = set(pd.read_csv(out)["stem"]) if out.exists() and out.stat().st_size else set()
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(["stem"] + [f"hr_std{c}" for c in range(4)] + [f"hr_mean{c}" for c in range(4)])
    n_new = 0
    for s in stems:
        if s in done:
            continue
        p = next((r / f"{s}.tif" for r in roots if (r / f"{s}.tif").exists()), None)
        if p is None:
            print("missing:", s); continue
        x = tifffile.imread(p).astype(np.float64)
        x = x if x.shape[-1] == 4 else np.moveaxis(x, 0, -1)
        w.writerow([s] + list(x.std((0, 1))) + list(x.mean((0, 1))))
        n_new += 1
    fh.close()
    print(f"{len(done) + n_new} tiles in {out}")


if __name__ == "__main__":
    main()
