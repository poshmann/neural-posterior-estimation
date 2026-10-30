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
                    NOT RECOMMENDED (29 Oct 2026): OLS shrinks the gain on poorly matching pairs
                    (regression dilution); 4-6 % of gains fall outside [0.5, 1.5]; negative reflectances.
--radiometry global one gain/offset per band for all tiles, from radiometry_fits.py
                    (--radiometry-json, e.g. the fit on the RQ3 train pairs = what arm 3 sees).
--radiometry quantile  RECOMMENDED: one monotone curve per band (global quantile mapping / CDF matching)
                    from the pooled pixels of the RQ3 train pairs (radiometry_fits.py --stems splits/train.csv,
                    "q_fakelr"/"q_s2" in --radiometry-json): a pixel at fakeLR's k-th percentile becomes S2's
                    k-th percentile. Reproduces the AOI darkening (close to multiplicative), monotone, no
                    extra negatives. Outside the fitted range: linear continuation with the edge slope.
--shift measured    ARM 2b (30 Oct 2026): replace the NPE shift magnitude by the shift MEASURED on the tile's own
                    real pair (shift_check.py, phase correlation fakeLR(PS) vs S2; --shift-csv). Needs
                    --source matched. sigma_b, sigma_n come from the tile's posterior sample whose r is closest
                    to the measured shift (approximates conditioning on r); phi, noise seed and radiometry are
                    drawn exactly as in arm 2, so arm 2 and 2b differ only in the shift magnitude.
                    Default output $RQ3_DATA/rq3_s2like_mshift. Reason: shift_check showed the NPE r is
                    ~0.7 PS px larger than the measured shift on the RQ3 train pairs (median 1.77 vs 0.89).
--family AOI        pool mode: only real tiles whose name starts with AOI_ (RQ3 train/val are all AOI_*;
                    tile_* scene pairs are blurrier, noisier, more shifted and brightness-matched).

Tiles: train + val + test stems of the RQ3 splits, PlanetScope cropped to 192 x 192 first (as for
arms 1 and 3). theta = 0 reproduces rq3_fake_lr exactly.  Output: $RQ3_DATA/rq3_s2like/<stem>.tif
(float32, H,W,C, the same TIFF layout as the other inputs) + rq3_s2like_recipes.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
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


def make_qmap(qx, qy):
    """Monotone map fakeLR value -> S2 value from matching quantile arrays (piecewise linear)."""
    qx, qy = np.asarray(qx, np.float64), np.asarray(qy, np.float64)
    keep = np.concatenate([[True], np.diff(qx) > 1e-9])          # strictly increasing x for np.interp
    qx, qy = qx[keep], np.maximum.accumulate(qy[keep])
    k = max(1, len(qx) // 200)                                    # edge slopes over the outer 0.5 %
    lo = np.clip((qy[k] - qy[0]) / max(qx[k] - qx[0], 1e-9), 0.0, 2.0)
    hi = np.clip((qy[-1] - qy[-1 - k]) / max(qx[-1] - qx[-1 - k], 1e-9), 0.0, 2.0)

    def f(v):
        out = np.interp(v, qx, qy)
        out = np.where(v < qx[0], qy[0] + (v - qx[0]) * lo, out)
        return np.where(v > qx[-1], qy[-1] + (v - qx[-1]) * hi, out).astype(np.float32)
    return f


def save(path, arr):
    tifffile.imwrite(path, np.ascontiguousarray(arr, dtype=np.float32), photometric="rgb",
                     planarconfig="contig", extrasamples=(2,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="npz from npe.py apply --samples-out")
    ap.add_argument("--source", choices=["pool", "matched"], default="pool")
    ap.add_argument("--radiometry", choices=["none", "pair", "global", "quantile"], default="none")
    ap.add_argument("--radiometry-json", default=None,
                    help="radiometry_global.json from radiometry_fits.py (for --radiometry global / quantile)")
    ap.add_argument("--family", choices=["", "AOI", "tile"], default="",
                    help="pool mode: only real tiles of this family (name starts with AOI_ / tile_)")
    ap.add_argument("--shift", choices=["npe", "measured"], default="npe",
                    help="measured = arm 2b: shift magnitude from shift_check.py (needs --source matched)")
    ap.add_argument("--shift-csv", nargs="+", default=None,
                    help="shift_check.csv file(s) covering all stems (e.g. shift_check_rq3_{train,val,test})")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--region", default="held_out",
                    help="pool mode: only use real tiles of this region (column in real/sentinel.csv); '' = all")
    ap.add_argument("--real-table", default=None, help="real/sentinel.csv (for --region filtering)")
    ap.add_argument("--out", default=None, help="default $RQ3_DATA/rq3_s2like")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    D, T = Path(os.environ["RQ3_DATA"]), Path(os.environ["RQ3_TEST"])
    splits_dir = Path(os.environ.get("RQ3_SPLITS", Path(__file__).resolve().parent / "splits"))
    out = Path(a.out or D / ("rq3_s2like_mshift" if a.shift == "measured" else "rq3_s2like"))
    out.mkdir(parents=True, exist_ok=True)
    meas = {}
    if a.shift == "measured":
        if a.source != "matched" or not a.shift_csv:
            raise SystemExit("--shift measured needs --source matched and --shift-csv")
        for f in a.shift_csv:
            for r in csv.DictReader(open(f)):
                ok = r["shift"] not in ("", "nan", "NaN")
                meas[r["tile"]] = (float(r["shift"]), float(r["peak"])) if ok else None
        n_none = sum(v is None for v in meas.values())
        print(f"measured shifts for {len(meas)} tiles from {len(a.shift_csv)} file(s); "
              f"{n_none} without a measurement (too much no-data) keep the arm-2 NPE draw")

    z = np.load(a.samples, allow_pickle=False)
    tiles, samples, names = list(z["tiles"]), z["samples"], list(z["names"])
    assert names == S.INFER, f"posterior is over {names}, simulator infers {S.INFER}"
    idx_of = {t: i for i, t in enumerate(tiles)}
    pool = list(range(len(tiles)))
    if a.source == "pool" and a.region and a.real_table:
        import pandas as pd
        reg = pd.read_csv(a.real_table).set_index("tile")["region"].to_dict()
        pool = [i for i, t in enumerate(tiles) if reg.get(t) == a.region]
    if a.source == "pool" and a.family:
        pool = [i for i in pool if str(tiles[i]).startswith(a.family + "_")]
    if not pool:
        raise SystemExit("no source tiles available")
    g_ab, qmaps = None, None
    if a.radiometry == "quantile":
        if not a.radiometry_json:
            raise SystemExit("--radiometry quantile needs --radiometry-json")
        gj = json.load(open(a.radiometry_json))
        qmaps = [make_qmap(gj["q_fakelr"][c], gj["q_s2"][c]) for c in range(4)]
        q_s2_ref = [np.asarray(gj["q_s2"][c]) for c in range(4)]
        print(f"quantile radiometry from {a.radiometry_json} ({gj.get('source')}, {gj['quantile_levels']} levels)")
    if a.radiometry == "global":
        if not a.radiometry_json:
            raise SystemExit("--radiometry global needs --radiometry-json")
        gj = json.load(open(a.radiometry_json))
        g_ab = list(zip(map(float, gj["gain"]), map(float, gj["offset"])))
        print(f"global radiometry from {a.radiometry_json} ({gj.get('source')}): "
              + "  ".join(f"b{c} {g:.3f}/{o:+.4f}" for c, (g, o) in enumerate(g_ab)))
    print(f"posterior samples: {len(tiles)} real tiles x {samples.shape[1]}; source={a.source} "
          f"(pool of {len(pool)}{', family ' + a.family if a.family else ''}); radiometry={a.radiometry}")

    jobs = []
    for sp in a.splits:
        root = T if sp == "test" else D
        for r in csv.DictReader(open(splits_dir / f"{sp}.csv")):
            jobs.append((sp, r["stem"], root))
    if a.limit:
        jobs = jobs[: a.limit]

    rec_path = out.parent / f"{out.name}_recipes.csv"
    cols = ["split", "stem", "source_tile", "sample", "sigma_b", "r", "phi", "sigma_n", "r_npe_draw", "peak"] + \
           [f"{k}{c}" for c in range(4) for k in ("gain", "offset")]
    fh = open(rec_path, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    ab_cache = {}
    neg_px = all_px = 0
    chk = [[] for _ in range(4)]
    chk_real = [[] for _ in range(4)]
    for n, (sp, stem, root) in enumerate(jobs, 1):
        rng = np.random.default_rng(zlib.crc32(f"arm2|{stem}".encode()))
        if a.source == "matched":
            if stem not in idx_of:
                raise SystemExit(f"--source matched: no posterior for {stem}")
            src = idx_of[stem]
        else:
            src = pool[rng.integers(len(pool))]
        k = int(rng.integers(samples.shape[1]))          # always drawn, so phi and noise match arm 2
        r_npe, peak = float(samples[src, k, S.INFER.index("r")]), ""
        if a.shift == "measured":
            if stem not in meas:
                raise SystemExit(f"--shift measured: no measured shift for {stem}")
            if meas[stem] is not None:
                r_meas, peak = meas[stem]
                k = int(np.argmin(np.abs(samples[src, :, S.INFER.index("r")] - r_meas)))
            else:
                r_meas, peak = r_npe, "none"            # no measurement: fall back to the arm-2 draw
        th = dict(zip(S.INFER, map(float, samples[src, k])))
        if a.shift == "measured":
            th["r"] = r_meas
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
        elif a.radiometry == "global":
            ab = g_ab
            deg = np.stack([ab[c][0] * deg[..., c] + ab[c][1] for c in range(4)], axis=2)
        elif a.radiometry == "quantile":
            deg = np.stack([qmaps[c](deg[..., c]) for c in range(4)], axis=2)
        real_p = root / "sr_60_200" / f"{stem}.tif"          # real S2 of the SAME tile, for the final check
        real = to_hwc(tifffile.imread(real_p))[:SIZE, :SIZE, :4] if real_p.exists() else None
        for c in range(4):
            chk[c].append(deg[..., c].ravel()[::37])
            if real is not None:
                chk_real[c].append(real[..., c].ravel()[::37])
        neg_px += int((deg < 0).sum()); all_px += deg.size
        save(out / f"{stem}.tif", deg)
        w.writerow({"split": sp, "stem": stem, "source_tile": tiles[src], "sample": k,
                    "sigma_b": theta[0], "r": theta[1], "phi": theta[2], "sigma_n": theta[3],
                    "r_npe_draw": r_npe, "peak": peak,
                    **{f"gain{c}": ab[c][0] for c in range(4)}, **{f"offset{c}": ab[c][1] for c in range(4)}})
        if n % 500 == 0 or n == len(jobs):
            fh.flush(); print(f"  {n}/{len(jobs)}", flush=True)
    fh.close()
    print(f"negative pixel values: {neg_px} of {all_px} ({100 * neg_px / max(all_px, 1):.3f} %) -- not clipped"
          "   (real AOI S2 train tiles: ~0.06 %)")
    lv = [1, 5, 25, 50, 75, 95, 99]
    print("pixel quantiles, arm 2 output vs REAL Sentinel-2 of the same tiles (should be close):")
    for c in range(4):
        v = np.concatenate(chk[c])
        line = f"  b{c} arm2    p" + " p".join(f"{l}={np.percentile(v, l):.4f}" for l in lv)
        if chk_real[c]:
            r = np.concatenate(chk_real[c])
            line += f"\n     real S2 p" + " p".join(f"{l}={np.percentile(r, l):.4f}" for l in lv)
            line += f"\n     ratio   " + "  ".join(f"{np.percentile(v, l) / max(np.percentile(r, l), 1e-6):.2f}"
                                                for l in lv)
        print(line)
    if a.shift == "measured":
        import pandas as pd
        rc = pd.read_csv(rec_path)
        print(f"shift used (measured) p10/50/90 {np.percentile(rc.r, [10, 50, 90]).round(2).tolist()}  vs arm-2 NPE draw "
              f"{np.percentile(rc.r_npe_draw, [10, 50, 90]).round(2).tolist()}  [PS px]")
    print(f"-> {out}  ({len(jobs)} tiles)  recipes: {rec_path}")


if __name__ == "__main__":
    main()
