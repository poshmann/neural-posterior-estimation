#!/usr/bin/env python3
"""
npe.py -- RQ1: neural posterior estimation of degradation parameters from fingerprints.

    python npe.py train  --train sims/train*.csv --val sims/val*.csv --out npe_runs/v1
    python npe.py train  ... --grid            # small hyper-parameter search, picked by val log-prob
    python npe.py apply  --run npe_runs/v1 --fingerprints real_s2.csv --out posterior_s2.csv

What it does
  input  x     = fingerprint (the 7 metrics, or whatever FEATURES are present), z-scored with the
                 TRAINING-simulation mean/sd (stored in the run, reused for real data)
  target theta = simulator.INFER = (sigma_b, r, sigma_n); prior = the simulator's uniform box
  model        = conditional normalising flow (sbi NPE), an ENSEMBLE of K flows (different seeds),
                 mixture posterior -> guards against over-confidence
  baseline     = rejection ABC: the k nearest training simulations in z-scored fingerprint space

Validation on held-out simulations (different tiles from training):
  * log p(theta_true | x)         higher = better (NPE only)
  * SBC ranks per parameter        uniform  <=> calibrated (KS p-value, histograms)
  * expected coverage              nominal central-interval level vs empirical hit rate
  * contraction = 1 - post_sd^2/prior_sd^2 per parameter   ~0: not identifiable, ~1: pinned down
  * RMSE of the posterior mean     vs the prior mean (= knowing nothing)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator as S  # noqa: E402

FEATURES = ["PSNR", "SSIM", "CWSSIM", "LPIPS_AVG", "DISTS_AVG", "MSSSIM_AVG", "CLIPcos_AVG"]
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


# ----------------------------------------------------------------------------- data
def load(patterns) -> pd.DataFrame:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no files match {patterns}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "r" not in df.columns and {"dx", "dy"} <= set(df.columns):      # older records
        df["r"] = np.hypot(df["dx"], df["dy"])
    return df


def add_context(df, context_csv):
    """Scene context: the tile's own theta=0 (fake-LR) fingerprint, columns b_<metric>.
    Computable for every real pair too, because the clean PlanetScope tile is known."""
    if not context_csv:
        return df
    files = sorted(glob.glob(context_csv))
    if not files:
        raise SystemExit(f"no context files match {context_csv}")
    ctx = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ctx = ctx[["stem" if "stem" in ctx.columns else "tile"] + [c for c in ctx.columns if c.startswith("b_")]]
    key = "stem" if "stem" in df.columns else "tile"
    ctx = ctx.rename(columns={"tile": "stem"})
    ctx["stem"] = ctx["stem"].astype(str).str.replace(".tif", "", regex=False)
    out = df.assign(_k=df[key].astype(str).str.replace(".tif", "", regex=False)) \
            .merge(ctx, left_on="_k", right_on="stem", how="left", suffixes=("", "_ctx"))
    miss = out[[c for c in ctx.columns if c.startswith("b_")]].isna().any(axis=1).sum()
    if miss:
        raise SystemExit(f"{miss} rows have no context fingerprint in {context_csv}")
    return out.drop(columns=["_k"] + [c for c in out.columns if c.endswith("_ctx")])


def features_present(df):
    feats = [f for f in FEATURES if f in df.columns and df[f].notna().all()]
    feats += [f"b_{f}" for f in FEATURES if f"b_{f}" in df.columns and df[f"b_{f}"].notna().all()]
    if len(feats) < 3:
        raise SystemExit(f"too few fingerprint columns: {feats}")
    return feats


def prior_box():
    from sbi.utils import BoxUniform
    lo = torch.tensor([S.PRIOR[k][0] for k in S.INFER], dtype=torch.float32)
    hi = torch.tensor([S.PRIOR[k][1] for k in S.INFER], dtype=torch.float32)
    return BoxUniform(lo, hi), lo.numpy(), hi.numpy()


# ----------------------------------------------------------------------------- training
def train_one(theta, x, prior, model, transforms, hidden, lr, batch, seed):
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    torch.manual_seed(seed); np.random.seed(seed)
    est = posterior_nn(model=model, hidden_features=hidden, num_transforms=transforms,
                       z_score_x="none", z_score_theta="independent")
    inf = NPE(prior=prior, density_estimator=est, show_progress_bars=False)
    inf.append_simulations(torch.as_tensor(theta, dtype=torch.float32),
                           torch.as_tensor(x, dtype=torch.float32))
    inf.train(training_batch_size=batch, learning_rate=lr, validation_fraction=0.1,
              stop_after_epochs=20, max_num_epochs=1000, show_train_summary=False)
    return inf.build_posterior()


class Ensemble:
    """Equal-weight mixture of K posteriors (written out, independent of sbi version)."""
    def __init__(self, posts):
        self.posts = posts

    def sample(self, n, x):
        k = len(self.posts)
        counts = np.random.multinomial(n, [1 / k] * k)
        parts = [p.sample((int(c),), x=x, show_progress_bars=False) for p, c in zip(self.posts, counts) if c]
        out = torch.cat(parts)
        return out[torch.randperm(len(out))]

    def log_prob(self, theta, x):
        lp = torch.stack([p.log_prob(theta, x=x) for p in self.posts])       # (K, n)
        return torch.logsumexp(lp, 0) - np.log(len(self.posts))


# ----------------------------------------------------------------------------- evaluation
def evaluate(sampler, logprob, theta_true, x, lo, hi, n_post=500, label=""):
    """sampler(x_row) -> (n_post, d) samples; logprob(theta, x_row) -> scalar or None."""
    d = theta_true.shape[1]
    ranks = np.zeros((len(x), d), int)
    hits = np.zeros((len(LEVELS), d))
    post_mean = np.zeros_like(theta_true)
    post_sd = np.zeros_like(theta_true)
    lps = []
    for i in range(len(x)):
        s = np.asarray(sampler(x[i], n_post))
        ranks[i] = (s < theta_true[i]).sum(0)
        post_mean[i], post_sd[i] = s.mean(0), s.std(0)
        for j, lev in enumerate(LEVELS):
            q_lo, q_hi = np.quantile(s, [(1 - lev) / 2, (1 + lev) / 2], axis=0)
            hits[j] += (theta_true[i] >= q_lo) & (theta_true[i] <= q_hi)
        if logprob is not None:
            lps.append(float(logprob(theta_true[i], x[i])))
    from scipy.stats import kstest
    prior_sd = (hi - lo) / np.sqrt(12)
    prior_mean = (hi + lo) / 2
    res = {"label": label, "n": int(len(x)),
           "mean_log_prob": float(np.mean(lps)) if lps else None,
           "params": {}}
    for k, name in enumerate(S.INFER):
        res["params"][name] = {
            "sbc_ks_p": float(kstest(ranks[:, k] / n_post, "uniform").pvalue),
            "coverage": {f"{lev:.2f}": float(h) for lev, h in zip(LEVELS, hits[:, k] / len(x))},
            "contraction": float(1 - np.mean(post_sd[:, k] ** 2) / prior_sd[k] ** 2),
            "rmse_post_mean": float(np.sqrt(np.mean((post_mean[:, k] - theta_true[:, k]) ** 2))),
            "rmse_prior_mean": float(np.sqrt(np.mean((prior_mean[k] - theta_true[:, k]) ** 2))),
        }
    res["_ranks"] = ranks
    return res


def report(results, path):
    lines = []
    for r in results:
        lines.append(f"== {r['label']}  (n = {r['n']} held-out simulations)")
        if r["mean_log_prob"] is not None:
            lines.append(f"   mean log p(theta_true | x) = {r['mean_log_prob']:.3f}")
        lines.append(f"   {'param':8s} {'SBC KS p':>9s} {'cov50':>6s} {'cov90':>6s} {'cov95':>6s} "
                     f"{'contract':>9s} {'RMSE post':>10s} {'RMSE prior':>11s}")
        for name, p in r["params"].items():
            c = p["coverage"]
            lines.append(f"   {name:8s} {p['sbc_ks_p']:9.3f} {c['0.50']:6.2f} {c['0.90']:6.2f} {c['0.95']:6.2f} "
                         f"{p['contraction']:9.2f} {p['rmse_post_mean']:10.4f} {p['rmse_prior_mean']:11.4f}")
        lines.append("")
    txt = "\n".join(lines)
    Path(path).write_text(txt)
    print(txt)


def plots(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = len(S.INFER)
    fig, ax = plt.subplots(2, d, figsize=(4 * d, 7))
    for r, col in zip(results, ["#1f6feb", "#d1495b"]):
        for k, name in enumerate(S.INFER):
            ax[0, k].hist(r["_ranks"][:, k], bins=20, histtype="step", lw=2, color=col, label=r["label"])
            cov = [r["params"][name]["coverage"][f"{lev:.2f}"] for lev in LEVELS]
            ax[1, k].plot(LEVELS, cov, "o-", color=col, label=r["label"])
    for k, name in enumerate(S.INFER):
        ax[0, k].set_title(f"SBC ranks: {name}  (flat = calibrated)")
        ax[1, k].plot([0, 1], [0, 1], "k--", lw=1)
        ax[1, k].set_title(f"coverage: {name}")
        ax[1, k].set_xlabel("nominal level"); ax[1, k].set_ylabel("empirical")
    ax[0, 0].legend(fontsize=8); ax[1, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130)


# ----------------------------------------------------------------------------- commands
def cmd_train(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr, va = add_context(load(a.train), a.context), add_context(load(a.val), a.context)
    shared = set(tr.stem) & set(va.stem)
    if shared:
        raise SystemExit(f"{len(shared)} tiles appear in both train and val simulations")
    feats = features_present(tr)
    mu, sd = tr[feats].mean().to_numpy(), tr[feats].std().to_numpy()
    xt = ((tr[feats].to_numpy() - mu) / sd).astype(np.float32)
    xv = ((va[feats].to_numpy() - mu) / sd).astype(np.float32)
    tt, tv = tr[S.INFER].to_numpy(np.float32), va[S.INFER].to_numpy(np.float32)
    prior, lo, hi = prior_box()
    print(f"train {len(tr)} sims / {tr.stem.nunique()} tiles | val {len(va)} / {va.stem.nunique()} tiles | "
          f"features {feats}")

    grid = [dict(model=m, transforms=t, hidden=h) for m in ("maf", "nsf") for t in (3, 5) for h in (64, 128)] \
        if a.grid else [dict(model=a.model, transforms=a.transforms, hidden=a.hidden)]
    sel = va.sample(n=min(len(va), 300), random_state=0).index      # val subset for selecting
    best = None
    for g in grid:
        t0 = time.time()
        post = train_one(tt, xt, prior, lr=a.lr, batch=a.batch, seed=0, **g)
        sel_pos = [va.index.get_loc(i) for i in sel]
        lp = float(np.mean([float(post.log_prob(torch.as_tensor(tv[i:i + 1]), x=torch.as_tensor(xv[i])))
                            for i in sel_pos]))
        print(f"  {g}  val log-prob {lp:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        if best is None or lp > best[0]:
            best = (lp, g, post)
    g = best[1]
    print(f"selected {g}; training ensemble of {a.ensemble}", flush=True)
    posts = [best[2]] + [train_one(tt, xt, prior, lr=a.lr, batch=a.batch, seed=s, **g)
                         for s in range(1, a.ensemble)]
    ens = Ensemble(posts)

    def npe_sampler(xrow, n):
        return ens.sample(n, torch.as_tensor(xrow)).numpy()

    def npe_lp(th, xrow):
        return ens.log_prob(torch.as_tensor(th[None]), torch.as_tensor(xrow)).item()

    k = max(20, int(a.abc_frac * len(tr)))

    def abc_sampler(xrow, n):
        idx = np.argsort(((xt - xrow) ** 2).sum(1))[:k]
        return tt[np.random.choice(idx, n, replace=True)]

    ev = va.drop(index=sel) if len(va) > len(sel) + 50 else va           # evaluate on the rest
    xe = ((ev[feats].to_numpy() - mu) / sd).astype(np.float32)
    te = ev[S.INFER].to_numpy(np.float32)
    n_eval = min(len(ev), a.n_eval)
    res = [evaluate(npe_sampler, npe_lp, te[:n_eval], xe[:n_eval], lo, hi, label=f"NPE ensemble x{a.ensemble}"),
           evaluate(abc_sampler, None, te[:n_eval], xe[:n_eval], lo, hi, label=f"rejection ABC (k={k})")]
    report(res, out / "report.txt")
    plots(res, out / "calibration.png")
    torch.save({"posteriors": posts, "features": feats, "mu": mu, "sd": sd, "infer": S.INFER,
                "prior": {k_: S.PRIOR[k_] for k_ in S.INFER}, "hparams": g}, out / "ensemble.pt")
    json.dump({"hparams": g, "features": feats, "ensemble": a.ensemble, "n_train": len(tr),
               "results": [{k_: v for k_, v in r.items() if k_ != "_ranks"} for r in res]},
              open(out / "summary.json", "w"), indent=2)
    print(f"saved -> {out}")


def cmd_apply(a):
    run = torch.load(Path(a.run) / "ensemble.pt", weights_only=False)
    ens = Ensemble(run["posteriors"])
    df = add_context(pd.read_csv(a.fingerprints), a.context)
    x = ((df[run["features"]].to_numpy() - run["mu"]) / run["sd"]).astype(np.float32)
    rows, keep = [], []
    for i in range(len(df)):
        s = ens.sample(a.n_samples, torch.as_tensor(x[i])).numpy()
        if a.samples_out:
            keep.append(s[: a.keep].astype(np.float32))
        r = {"tile": df.iloc[i].get("tile", df.iloc[i].get("stem", i))}
        for k, name in enumerate(run["infer"]):
            q = np.quantile(s[:, k], [0.05, 0.5, 0.95])
            r.update({f"{name}_mean": s[:, k].mean(), f"{name}_sd": s[:, k].std(),
                      f"{name}_q05": q[0], f"{name}_q50": q[1], f"{name}_q95": q[2]})
        rows.append(r)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} posteriors -> {a.out}")
    if a.samples_out:
        np.savez_compressed(a.samples_out, tiles=np.array([str(r["tile"]) for r in rows]),
                            samples=np.stack(keep), names=np.array(run["infer"]))
        print(f"posterior samples ({len(keep)} tiles x {a.keep}) -> {a.samples_out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", nargs="+", required=True)
    t.add_argument("--val", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="nsf")
    t.add_argument("--transforms", type=int, default=5)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--ensemble", type=int, default=5)
    t.add_argument("--grid", action="store_true")
    t.add_argument("--abc-frac", type=float, default=0.02)
    t.add_argument("--n-eval", type=int, default=400)
    t.add_argument("--context", default=None, help="CSV(s, glob ok) with stem + b_<metric>: theta=0 fingerprint per tile")
    p = sub.add_parser("apply")
    p.add_argument("--run", required=True)
    p.add_argument("--fingerprints", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--context", default=None)
    p.add_argument("--samples-out", default=None, help=".npz with posterior samples per tile (for build_arm2.py)")
    p.add_argument("--keep", type=int, default=200, help="samples per tile kept in --samples-out")
    a = ap.parse_args()
    {"train": cmd_train, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
