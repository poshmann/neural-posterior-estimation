#!/usr/bin/env python3
"""
evaluate_rq3.py -- score super-resolved test tiles against PlanetScope, per tile, for every arm.

    python evaluate_rq3.py \
        --hr "$RQ3_TEST/hr_200" \
        --arm input_s2="$RQ3_TEST/sr_60_200" \
        --arm arm1_fakelr="$RQ3_RUNS/rq3_arm1_fakelr/eval_s2" \
        --arm arm3_s2="$RQ3_RUNS/rq3_arm3_s2/eval_s2" \
        --reference arm3_s2 --device cuda --out "$RQ3_RUNS/rq3_eval"

    --arm NAME=FOLDER[:m11]   FOLDER holds <stem>.tif; ':m11' if the tiles are in [-1, 1] (old WaveCM outputs)
    --reference NAME          paired differences are reported against this arm

Two families of scores (both on the top-left 192 x 192 crop, the area the models see):
  P ruler  (the paper's fingerprint, per-band min-max of each image first -> scale-free)
           PSNR, SSIM, CWSSIM, LPIPS_AVG, DISTS_AVG, MSSSIM_AVG, CLIPcos_AVG
  absolute (reflectance, no rescaling -> also punishes brightness / colour errors)
           absPSNR (data range 1.0), absSSIM (per band, data range 1.0), absMAE

Outputs in --out:
  per_tile.csv        one row per (arm, tile)
  summary.csv/.txt    mean and 95% bootstrap CI per arm and metric
  paired.csv/.txt     arm - reference: mean paired difference, 95% bootstrap CI, Wilcoxon p, wins/128
  metrics.png         per-metric distributions (one dot per tile), arms side by side
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator as S  # noqa: E402

SIZE = 192
P_METRICS = S.METRICS
ABS_METRICS = ["absPSNR", "absSSIM", "absMAE"]
HIGHER_BETTER = {"PSNR": True, "SSIM": True, "CWSSIM": True, "LPIPS_AVG": False, "DISTS_AVG": False,
                 "MSSSIM_AVG": True, "CLIPcos_AVG": False, "absPSNR": True, "absSSIM": True, "absMAE": False}


def to_hwc(a):
    a = np.asarray(a, dtype=np.float32)
    return a if a.shape[-1] == 4 else np.moveaxis(a, 0, -1)


def load(path, m11=False):
    a = to_hwc(tifffile.imread(path))[:SIZE, :SIZE, :4]
    return (a + 1) / 2 if m11 else a


def abs_scores(hr, sr):
    from skimage.metrics import structural_similarity as ssim
    mse = float(np.mean((sr - hr) ** 2))
    return {"absPSNR": float("inf") if mse == 0 else 10 * np.log10(1.0 / mse),
            "absSSIM": float(np.mean([ssim(hr[..., c], sr[..., c], data_range=1.0, gaussian_weights=True,
                                           sigma=1.5, use_sample_covariance=False) for c in range(4)])),
            "absMAE": float(np.mean(np.abs(sr - hr)))}


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-perceptual", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    arms = {}
    for spec in a.arm:
        name, folder = spec.split("=", 1)
        m11 = folder.endswith(":m11")
        arms[name] = (Path(folder[:-4] if m11 else folder), m11)
    hr_dir = Path(a.hr)
    stems = sorted(set.intersection(*[{p.stem for p in f.glob("*.tif") if not p.name.startswith(".")}
                                      for f, _ in arms.values()]) & {p.stem for p in hr_dir.glob("*.tif")})
    if not stems:
        raise SystemExit("no tile is present in every arm folder and the HR folder")
    for name, (f, _) in arms.items():
        n = len([p for p in f.glob("*.tif") if not p.name.startswith(".")])
        if n != len(stems):
            print(f"note: {name} has {n} tiles, {len(stems)} are common to all arms -> scoring those")
    print(f"{len(stems)} tiles x {len(arms)} arms: {list(arms)}")

    fp = S.Fingerprinter(device=a.device, perceptual=not a.no_perceptual)
    pm = P_METRICS if not a.no_perceptual else ["PSNR", "SSIM", "CWSSIM"]
    rows = []
    for i, stem in enumerate(stems, 1):
        hr = load(hr_dir / f"{stem}.tif")
        for name, (f, m11) in arms.items():
            sr = load(f / f"{stem}.tif", m11)
            r = fp(hr, sr)
            rows.append({"arm": name, "stem": stem, **{k: float(r[k]) for k in pm}, **abs_scores(hr, sr)})
        if i % 16 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_tile.csv", index=False)
    metrics = pm + ABS_METRICS

    # summary
    srows, lines = [], [f"{len(stems)} test tiles; mean [95% bootstrap CI];  higher-better: "
                        + ", ".join(m for m in metrics if HIGHER_BETTER[m])]
    for name in arms:
        d = df[df.arm == name]
        line = f"{name:16s}"
        for m in metrics:
            mu, lo, hi = boot_ci(d[m])
            srows.append({"arm": name, "metric": m, "mean": mu, "ci_lo": lo, "ci_hi": hi})
            line += f" | {m} {mu:.4f} [{lo:.4f}, {hi:.4f}]"
        lines.append(line)
    pd.DataFrame(srows).to_csv(out / "summary.csv", index=False)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # paired differences against the reference arm
    if a.reference:
        from scipy.stats import wilcoxon
        ref = df[df.arm == a.reference].set_index("stem")
        prow, plines = [], [f"paired differences: arm - {a.reference} (same tiles); 'better' respects metric direction"]
        for name in arms:
            if name == a.reference:
                continue
            d = df[df.arm == name].set_index("stem").loc[ref.index]
            for m in metrics:
                diff = (d[m] - ref[m]).to_numpy()
                mu, lo, hi = boot_ci(diff)
                better = diff > 0 if HIGHER_BETTER[m] else diff < 0
                p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                prow.append({"arm": name, "metric": m, "mean_diff": mu, "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_p": p, "tiles_better": int(better.sum()), "n": len(diff)})
                plines.append(f"{name:16s} {m:12s} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={p:.2g}  "
                              f"better on {int(better.sum())}/{len(diff)}")
        pd.DataFrame(prow).to_csv(out / "paired.csv", index=False)
        (out / "paired.txt").write_text("\n".join(plines) + "\n")
        print("\n".join(plines))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 5
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axs.flat, metrics):
        for k, name in enumerate(arms):
            v = df[df.arm == name][m].to_numpy()
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=6, alpha=0.4)
            ax.plot([k - 0.3, k + 0.3], [v.mean()] * 2, "k-", lw=2)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(list(arms), rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{m} ({'higher' if HIGHER_BETTER[m] else 'lower'} = better)", fontsize=9)
    for ax in list(axs.flat)[len(metrics):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "metrics.png", dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
