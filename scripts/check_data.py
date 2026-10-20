#!/usr/bin/env python3
"""Check that every tile the RQ3 splits need is on HAICORE and readable.

    source env.sh && python check_data.py            # HR + Sentinel-2 + test
    python check_data.py --with-fakelr                   # also rq3_fake_lr/ (after make_fake_lr.py)

Checks per tile: file exists, one 4-band image, >= 192 px, finite values.
Exit code 1 if anything is missing or broken.
"""
import argparse, csv, os, sys
import numpy as np
import tifffile


def stems(name):
    with open(os.path.join(os.environ["RQ3_SPLITS"], name)) as fh:
        return [r["stem"] for r in csv.DictReader(fh)]


def to_hwc(a):
    return a if a.shape[-1] == 4 else np.moveaxis(a, 0, -1)


def check(folder, wanted, label, full=False):
    bad, missing = [], []
    for i, s in enumerate(wanted):
        p = os.path.join(folder, s + ".tif")
        if not os.path.isfile(p):
            missing.append(s); continue
        try:
            with tifffile.TiffFile(p) as t:
                shp = t.series[0].shape
                if full or i % 20 == 0:          # values: every tile with --full, else every 20th
                    a = to_hwc(t.asarray())
                    if not np.isfinite(a).all(): raise ValueError("NaN/inf")
            hwc = shp if shp[-1] == 4 else (shp[1], shp[2], shp[0])
            if len(shp) != 3 or hwc[2] != 4 or hwc[0] < 192 or hwc[1] < 192:
                raise ValueError(f"shape {shp}")
        except Exception as e:
            bad.append((s, str(e)))
    status = "OK " if not (bad or missing) else "BAD"
    print(f"[{status}] {label:<28} {len(wanted) - len(bad) - len(missing):>5}/{len(wanted)} good"
          f" | missing {len(missing)} | broken {len(bad)}  ({folder})")
    for s in missing[:5]: print("      missing:", s)
    for s, e in bad[:5]: print("      broken :", s, e)
    return not (bad or missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-fakelr", action="store_true")
    ap.add_argument("--full", action="store_true", help="read every tile's values (slower)")
    a = ap.parse_args()
    D, T = os.environ["RQ3_DATA"], os.environ["RQ3_TEST"]
    trval = stems("train.csv") + stems("val.csv")
    test = stems("test.csv")
    ok = True
    ok &= check(os.path.join(D, "hr_200"), trval, "train+val PlanetScope", a.full)
    ok &= check(os.path.join(D, "sr_60_200"), trval, "train+val Sentinel-2", a.full)
    ok &= check(os.path.join(T, "hr_200"), test, "test PlanetScope", a.full)
    ok &= check(os.path.join(T, "sr_60_200"), test, "test Sentinel-2", a.full)
    if a.with_fakelr:
        ok &= check(os.path.join(D, "rq3_fake_lr"), trval + test, "fake-LR (train+val+test)", a.full)
    print("ALL GOOD" if ok else "PROBLEMS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
