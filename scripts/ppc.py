#!/usr/bin/env python3
"""
ppc.py -- posterior predictive check: can the simulator, at theta drawn from a tile's posterior,
reproduce that tile's REAL fingerprint?   (RQ2: is the forward model adequate for real Sentinel-2?)

    python ppc.py --samples post_s2_samples.npz --real real/sentinel.csv --draws 20 --device cuda --out ppc/

For each real tile: draw `--draws` theta from its posterior (phi uniform), simulate on the tile's own
PlanetScope image (native size, $RQ3_DATA or $RQ3_TEST/hr_200), fingerprint with the P ruler, and
locate the real value of every metric within the simulated ones (predictive percentile, 0..1).

Reading it
  * percentiles roughly uniform              -> the model can produce what real Sentinel-2 looks like
  * piled up at 0 or 1 for a metric          -> real pairs are systematically worse/better on that metric
                                                than ANY posterior degradation can explain (misspecified)
  * sanity check of the machinery: use held-out SIMULATIONS as "real" input -> must look uniform.
    (Not fake-LR: it sits exactly at theta = 0, the edge of the prior, so every posterior draw is a
    little more degraded and the real value lands at the top by construction.)
Outputs: ppc_per_tile.csv, ppc_summary.txt, ppc.png
"""
import argparse
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--real", required=True, help="csv with tile + the real fingerprint (e.g. real/sentinel.csv)")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--max-tiles", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-perceptual", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.samples)
    tiles, samples = list(z["tiles"]), z["samples"]
    assert list(z["names"]) == S.INFER
    real = pd.read_csv(a.real)
    real["tile"] = real["tile"].astype(str).str.replace(".tif", "", regex=False)
    real = real.set_index("tile")
    metrics = ["PSNR", "SSIM", "CWSSIM"] + ([] if a.no_perceptual else
                                            ["LPIPS_AVG", "DISTS_AVG", "MSSSIM_AVG", "CLIPcos_AVG"])
    roots = [Path(os.environ["RQ3_DATA"]) / "hr_200", Path(os.environ["RQ3_TEST"]) / "hr_200"]
    fp = S.Fingerprinter(device=a.device, perceptual=not a.no_perceptual)

    order = np.random.default_rng(0).permutation(len(tiles))[: a.max_tiles]
    rows = []
    for n, i in enumerate(order, 1):
        t = tiles[i]
        path = next((r / f"{t}.tif" for r in roots if (r / f"{t}.tif").exists()), None)
        if path is None or t not in real.index:
            continue
        hr = fp.A.load_tile(path)
        rng = np.random.default_rng(zlib.crc32(f"ppc|{t}".encode()))
        sims = []
        for k in rng.choice(samples.shape[1], a.draws, replace=False):
            th = dict(zip(S.INFER, samples[i, k]))
            theta = (th["sigma_b"], th["r"], rng.uniform(0, 2 * np.pi), th["sigma_n"])
            r = fp(hr, S.degrade(hr, theta, rng))
            sims.append([r[m] for m in metrics])
        sims = np.array(sims)
        row = {"tile": t}
        for j, m in enumerate(metrics):
            x = float(real.loc[t, m])
            # randomised percentile (ties broken uniformly) -> exactly uniform under a correct model
            below, ties = (sims[:, j] < x).sum(), (sims[:, j] == x).sum()
            row[f"pct_{m}"] = (below + rng.uniform() * (ties + 1)) / (len(sims) + 1)
            row[f"real_{m}"], row[f"simmean_{m}"] = x, float(sims[:, j].mean())
        rows.append(row)
        if n % 25 == 0:
            print(f"  {n}/{len(order)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "ppc_per_tile.csv", index=False)

    from scipy.stats import kstest
    lines = [f"posterior predictive check: {len(df)} tiles x {a.draws} draws (percentile of the real value)"]
    lines.append(f"{'metric':12s} {'KS p':>7s} {'<5%':>6s} {'>95%':>6s}  {'real mean':>10s} {'sim mean':>10s}")
    for m in metrics:
        p = df[f"pct_{m}"]
        lines.append(f"{m:12s} {kstest(p, 'uniform').pvalue:7.3f} {(p < .05).mean():6.2f} {(p > .95).mean():6.2f}  "
                     f"{df[f'real_{m}'].mean():10.4f} {df[f'simmean_{m}'].mean():10.4f}")
    lines.append("expected under a correct model: KS p not small, ~0.05 in each tail")
    (out / "ppc_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 2.6))
    for ax, m in zip(np.atleast_1d(axs), metrics):
        ax.hist(df[f"pct_{m}"], bins=10, range=(0, 1), color="#1f6feb")
        ax.axhline(len(df) / 10, color="k", ls="--", lw=1)
        ax.set_title(m, fontsize=9); ax.set_xlabel("predictive percentile", fontsize=8)
    fig.suptitle("flat = the simulator can reproduce the real fingerprints", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "ppc.png", dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
