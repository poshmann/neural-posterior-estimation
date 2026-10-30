#!/usr/bin/env python3
"""
rq2_analysis.py -- RQ2: what does the NPE say about REAL Sentinel-2 degradations?

Inputs (from `npe.py apply`):
    post_sentinel.csv   per-tile posterior summaries for real S2 vs PlanetScope
    post_fakelr.csv     the same for fake-LR (positive control: should sit at theta ~ 0)
    real/sentinel.csv   region (held_out / sim_train) and group per tile (prepare_real.py)

    python rq2_analysis.py --post-s2 rq2/post_sentinel.csv --post-fk rq2/post_fakelr.csv \
        --real real/sentinel.csv --out rq2/

Writes  rq2/summary.txt  and  rq2/rq2_posteriors.png
Reported on held-out tiles (AOIs/scenes that gave no NPE training simulation). Group-bootstrap
CIs (resampling AOIs/scenes, not tiles, because tiles of one scene are not independent).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulator import INFER, PRIOR

UNITS = {"sigma_b": "PS px", "r": "PS px", "sigma_n": "reflectance"}
LABEL = {"sigma_b": "blur σ_b", "r": "shift r", "sigma_n": "noise σ_n"}


def group_boot(df, col, stat=np.mean, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = {k: v[col].to_numpy() for k, v in df.groupby("group")}
    keys = np.array(list(g))
    vals = [stat(np.concatenate([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def describe(df, name, out):
    lines = [f"== {name}: {len(df)} tiles, {df.group.nunique()} groups"]
    lines.append(f"   {'param':8s} {'median of medians':>18s} {'mean [95% group CI]':>28s} "
                 f"{'IQR of medians':>18s} {'contraction':>12s} {'at edge':>8s} {'top 10%':>8s}")
    for p in INFER:
        lo, hi = PRIOR[p]
        prior_var = (hi - lo) ** 2 / 12
        med = df[f"{p}_q50"]
        ci = group_boot(df, f"{p}_q50")
        contr = 1 - np.mean(df[f"{p}_sd"] ** 2) / prior_var
        edge = np.mean(df[f"{p}_q95"] < lo + 0.1 * (hi - lo))   # posterior entirely in the lowest 10% of the prior
        top = np.mean(med > hi - 0.1 * (hi - lo))                   # posterior median in the highest 10% of the prior
        q1, q3 = med.quantile([0.25, 0.75])
        lines.append(f"   {p:8s} {med.median():18.4g} {med.mean():10.4g} [{ci[0]:.4g}, {ci[1]:.4g}]"
                     f"{'':3s}{q1:8.4g}-{q3:<8.4g} {contr:12.2f} {edge:8.2f} {top:8.2f}")
    for c in [c for c in df.columns if c.startswith("corr_")]:
        lines.append(f"   {c}: median {df[c].median():+.2f}, "
                     f"share < -0.3: {np.mean(df[c] < -0.3):.2f}, share > +0.3: {np.mean(df[c] > 0.3):.2f}")
    out += lines
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-s2", required=True)
    ap.add_argument("--post-fk", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(a.real)[["tile", "group", "region"]]
    s2 = pd.read_csv(a.post_s2).merge(meta, on="tile")
    fk = pd.read_csv(a.post_fk).merge(meta, on="tile")

    txt = ["RQ2 -- NPE posteriors on real pairs (contraction = 1 - mean post var / prior var;",
           "'at edge' = share of tiles whose 95% quantile lies in the lowest 10% of the prior;",
           "'top 10%' = share of tiles whose posterior median lies in the highest 10% of the prior)", ""]
    for region in ("held_out", "sim_train"):
        txt = describe(fk[fk.region == region], f"fake-LR control, {region}", txt)
        txt = describe(s2[s2.region == region], f"real Sentinel-2, {region}", txt)
        txt.append("")
    # tile families: AOI_* and tile_* differ in brightness matching and blur/shift (see handoff 30 Oct)
    fam = lambda d: np.where(d.tile.astype(str).str.startswith("AOI"), "AOI", "tile_")
    s2["family"], fk["family"] = fam(s2), fam(fk)
    txt.append("-- held-out tiles split by family (AOI_* vs tile_*)")
    for f in ("AOI", "tile_"):
        txt = describe(fk[(fk.region == "held_out") & (fk.family == f)], f"fake-LR control, held_out, {f}", txt)
        txt = describe(s2[(s2.region == "held_out") & (s2.family == f)], f"real Sentinel-2, held_out, {f}", txt)
        txt.append("")
    (out_dir / "summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))

    # ---------------------------------------------------------------- figure (held-out tiles)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_S2, C_FK, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False})
    h2, hf = s2[s2.region == "held_out"], fk[fk.region == "held_out"]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.1), constrained_layout=True)
    for i, p in enumerate(INFER):
        lo, hi = PRIOR[p]
        bins = np.linspace(lo, hi, 41)
        ax[i].hist(hf[f"{p}_q50"], bins=bins, color=C_FK, alpha=0.85, label="fake-LR (control)",
                   edgecolor="#fcfcfb", linewidth=0.5)
        ax[i].hist(h2[f"{p}_q50"], bins=bins, color=C_S2, alpha=0.85, label="real Sentinel-2",
                   edgecolor="#fcfcfb", linewidth=0.5)
        ax[i].set_xlabel(f"posterior median {LABEL[p]} [{UNITS[p]}]")
        ax[i].set_xlim(lo, hi)
        ax[i].grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax[i].set_axisbelow(True)
    ax[0].set_ylabel("tiles")
    ax[0].legend(frameon=False, loc="upper right")
    if "corr_sigma_b_sigma_n" in h2:
        ax[3].hist(h2["corr_sigma_b_sigma_n"], bins=np.linspace(-1, 1, 41), color=C_S2,
                   edgecolor="#fcfcfb", linewidth=0.5)
        ax[3].axvline(0, color=INK2, linewidth=1)
        ax[3].set_xlabel("posterior corr(σ_b, σ_n), real S2")
        ax[3].grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax[3].set_axisbelow(True)
    fig.suptitle(f"Held-out tiles: real Sentinel-2 (n={len(h2)}) vs fake-LR control (n={len(hf)})",
                 color=INK, fontsize=10)
    fig.savefig(out_dir / "rq2_posteriors.png", dpi=200, facecolor="#fcfcfb")
    print(f"figure -> {out_dir / 'rq2_posteriors.png'}")


if __name__ == "__main__":
    main()
