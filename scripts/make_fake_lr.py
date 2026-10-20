#!/usr/bin/env python3
"""Regenerate the RQ3 fake-LR tiles on HAICORE, identical to the ones built on the Mac
(~/Desktop/neural-posterior-estimation/Training Data/*/fake_lr), and verify them.

Recipe (from Test/Test_fakelr/make_fakelr_testset.py):
    PlanetScope HR, top-left crop to 192x192
    -> bicubic down to 60x60 -> bicubic back up to 192x192
    skimage.transform.resize(order=3, mode="reflect", anti_aliasing=True, preserve_range=True)
NOTE: the original WaveCM fake_lr/ folder used 200->60->200 on the 200-px tiles and lacks
96 of our tiles, so it is NOT used.

    source env.sh && python make_fake_lr.py        # writes $RQ3_DATA/rq3_fake_lr/
Output: one float32 (H,W,C) tile per stem, same TIFF layout as the PlanetScope tiles.
Verification: compares each tile with splits/fake_lr_checksums.csv (made on the Mac).
"""
import csv, hashlib, os, sys
import numpy as np
import tifffile
from skimage.transform import resize

SIZE, DOWN = 192, 60


def bicubic(a, n):
    return resize(a, (n, n, a.shape[2]), order=3, mode="reflect",
                  anti_aliasing=True, preserve_range=True).astype(np.float32)


def to_hwc(a):
    return a if a.shape[-1] == 4 else np.moveaxis(a, 0, -1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output folder (default $RQ3_DATA/rq3_fake_lr)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tiles (for testing)")
    args = ap.parse_args()
    D, T, S = os.environ["RQ3_DATA"], os.environ["RQ3_TEST"], os.environ["RQ3_SPLITS"]
    out = args.out or os.path.join(D, "rq3_fake_lr")
    os.makedirs(out, exist_ok=True)
    ref = {r["stem"]: r for r in csv.DictReader(open(os.path.join(S, "fake_lr_checksums.csv")))}
    jobs = []
    for split, root in (("train", D), ("val", D), ("test", T)):
        for r in csv.DictReader(open(os.path.join(S, f"{split}.csv"))):
            jobs.append((r["stem"], os.path.join(root, "hr_200", r["stem"] + ".tif")))
    if args.limit:
        jobs = jobs[:args.limit // 2] + jobs[-(args.limit - args.limit // 2):]   # some train + some test
    exact = close = wrong = 0
    worst = 0.0
    for i, (stem, hr_path) in enumerate(jobs, 1):
        hr = to_hwc(tifffile.imread(hr_path)).astype(np.float32)[:SIZE, :SIZE, :]
        fk = np.ascontiguousarray(bicubic(bicubic(hr, DOWN), SIZE), dtype=np.float32)
        tifffile.imwrite(os.path.join(out, stem + ".tif"), fk, photometric="rgb",
                         planarconfig="contig", extrasamples=(2,))
        r = ref[stem]
        if hashlib.md5(fk.tobytes()).hexdigest() == r["md5"]:
            exact += 1
        else:  # different library builds can differ in the last float bits
            d = max(abs(float(fk.mean(dtype=np.float64)) - float(r["mean"])),
                    abs(float(fk.std(dtype=np.float64)) - float(r["std"])))
            worst = max(worst, d)
            if d < 1e-6: close += 1
            else: wrong += 1
        if i % 500 == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)}  identical {exact}  float-noise only {close}  DIFFERENT {wrong}", flush=True)
    print(f"done -> {out}  (largest mean/std difference {worst:.2e})")
    if wrong:
        print("!! Some tiles differ from the Mac version -- do NOT train arm 1; "
              "upload the Mac fake_lr folders instead (see README.md).")
        sys.exit(1)


if __name__ == "__main__":
    main()
