#!/usr/bin/env python3
"""
prepare_real.py -- real fingerprints for RQ2, ready for `npe.py apply`.

The paper already scored 3,000 real pairs (results_3000/, P ruler, native tile size):
    sentinel_per_tile.csv   Sentinel-2 vs PlanetScope      -> what we want the posterior for
    fakelr_per_tile.csv     fake-LR   vs PlanetScope      -> identical to the simulator at theta = 0
                                                             (checked: PSNR/SSIM/CW-SSIM equal to 1e-14)
So the scene context b_<metric> of every real tile is simply its fake-LR row. No new computation.

    python prepare_real.py --results ~/Desktop/super-resolution-degradation/results_3000 --out real/

Writes
    real/sentinel.csv   tile, 7 metrics, b_<7 metrics>, group, region   (input for npe.py apply)
    real/fakelr.csv     the same for fake-LR = POSITIVE CONTROL: its posterior must sit at theta ~ 0
region = 'sim_train' if the tile's AOI/scene also provided NPE training simulations (RQ3 train split),
         else 'held_out'.  Report RQ2 on 'held_out' tiles; compare both as a check.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

M = ["PSNR", "SSIM", "CWSSIM", "LPIPS_AVG", "DISTS_AVG", "MSSSIM_AVG", "CLIPcos_AVG"]
HERE = Path(__file__).resolve().parent


def group(stem: str) -> str:            # same rule as build_dataset.py
    s = stem.split("_")
    return f"AOI_{s[1]}" if s[0] == "AOI" else f"scene_{s[2]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder with sentinel_per_tile.csv and fakelr_per_tile.csv")
    ap.add_argument("--splits", default=str(HERE / "splits"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    s2 = pd.read_csv(Path(a.results) / "sentinel_per_tile.csv")
    fk = pd.read_csv(Path(a.results) / "fakelr_per_tile.csv")
    for d in (s2, fk):
        d["stem"] = d["tile"].astype(str).str.replace(".tif", "", regex=False)
    ctx = fk[["stem"] + M].rename(columns={m: f"b_{m}" for m in M})
    train_groups = {group(r["stem"]) for r in csv.DictReader(open(Path(a.splits) / "train.csv"))}
    split_of = {}
    for sp in ("train", "val", "test"):
        for r in csv.DictReader(open(Path(a.splits) / f"{sp}.csv")):
            split_of[r["stem"]] = sp

    for name, d in (("sentinel", s2), ("fakelr", fk)):
        t = d[["stem"] + M].merge(ctx, on="stem", how="inner")
        t["group"] = t["stem"].map(group)
        t["region"] = ["sim_train" if g in train_groups else "held_out" for g in t["group"]]
        t["rq3_split"] = t["stem"].map(split_of).fillna("-")
        t = t.dropna(subset=M + [f"b_{m}" for m in M])
        t.insert(0, "tile", t["stem"])
        t.drop(columns=["stem"]).to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(t)} tiles | held_out {int((t.region == 'held_out').sum())} "
              f"({t[t.region == 'held_out'].group.nunique()} groups) | sim_train {int((t.region == 'sim_train').sum())} "
              f"| in RQ3 splits: {t.rq3_split.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
