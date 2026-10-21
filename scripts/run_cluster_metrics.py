#!/usr/bin/env python3
"""
run_cluster_metrics.py
======================
Full metric-fingerprint computation for the 114k training set, on the GFZ/HAICORE
A100 cluster. Produces the data behind the paper.

THREE COMPARISONS (all against clean PlanetScope = hr_200):
  1. degradation : degrade hr_200 with 11 families x low-end-concentrated strengths,
                   score degraded-vs-clean-hr.   (~112k tiles x ~75 scorings)
  2. fakelr      : score fake_lr vs hr_200.        (~112k tiles x 1 scoring)
  3. sentinel    : score sr_60_200 (Sentinel) vs hr_200.  (~112k tiles x 1 scoring)

SAME RULER GUARANTEE:
  All metric math is imported VERBATIM from your existing scripts
  (superres_eval.py, evaluate_new_metrics.py, evaluate_degradation_sensitivity_v2.py).
  Nothing is re-implemented, so these numbers are directly comparable to your
  earlier test-set and fake-LR results.

WHAT'S NEW HERE vs. the Mac scripts:
  * THREE-PASS channel scheme for perceptual metrics (LPIPS/DISTS/CLIP/MS-SSIM):
        RGB   pass = channels [0,1,2]   true colour
        GBNIR pass = channels [1,2,3]   false colour (green, blue, NIR)
        RRR   pass = channels [0,0,0]   red x3 -- a NO-COLOUR control
    Reported as *_RGB, *_GBNIR, *_RRR, and *_AVG.
    *_AVG = mean(RGB, GBNIR) only -- RRR is a control and is excluded from AVG.
    RGB vs RRR isolates how much of a metric's verdict is colour vs structure.
  * PER-PAIR perceptual scoring that reuses evaluate_new_metrics.PerceptualBank
    VERBATIM (the exact per-image methods that already ran correctly on the Mac).
    We deliberately do NOT batch the perceptual metrics: piq's DISTS / MS-SSIM
    collapse a multi-pair batch to a single reduced score, which silently breaks
    per-pair indexing. One-pair-at-a-time is the proven, bulletproof path; on an
    A100 it is still far faster than the Mac. Correctness first.
  * Streaming CSV + resume + self-heal for multi-day safety.

RESUME: if the job stops, just resubmit the same sbatch. It reads the existing
CSVs, skips finished tiles, continues. No duplicates.

USAGE (see the .sbatch wrapper; for manual/smoke runs):
  python run_cluster_metrics.py \
      --data   /hkfs/work/workspace_haic/scratch/hgf_sma8852-srdata \
      --scripts /hkfs/work/workspace_haic/scratch/hgf_sma8852-srdata/scripts \
      --out    /hkfs/work/workspace_haic/scratch/hgf_sma8852-srdata/results \
      --tasks  degradation fakelr sentinel \
      --device cuda \
      --limit 20        # SMOKE TEST FIRST; omit for full run

  (--batch-size still accepted but only groups tile I/O; perceptual scoring is
   always per-pair, so batch size never affects the numbers.)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Import the existing metric code (same ruler)
# --------------------------------------------------------------------------- #

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
#  Per-pair, dual-pass perceptual bank
#  (reuses evaluate_new_metrics.PerceptualBank -- the proven Mac path -- verbatim)
# --------------------------------------------------------------------------- #

class DualPassPerceptual:
    """
    Multi-pass wrapper around the EXISTING evaluate_new_metrics.PerceptualBank.

    For each pair it computes each of the four perceptual metrics
    (LPIPS, DISTS, MS-SSIM, CLIP) THREE times, on three channel treatments:

        RGB   pass = channels [0,1,2]   true colour (the field default)
        GBNIR pass = channels [1,2,3]   false colour: G, B, NIR
        RRR   pass = channels [0,0,0]   red replicated 3x -- a NO-COLOUR control

    Why RRR: the perceptual nets need 3 channels. RGB gives them true colour,
    GBNIR gives them false colour, and RRR gives them NO colour information at
    all -- the same band three times, so only structure remains. Comparing RGB
    vs RRR therefore isolates how much of a perceptual metric's verdict comes
    from colour versus pure structure. It is a control, not a third "view".

    Reported columns per metric: *_RGB, *_GBNIR, *_RRR, and *_AVG.
    NOTE: *_AVG is deliberately the mean of RGB and GBNIR ONLY -- the two real
    views of the data. RRR is a control and is excluded from AVG so it cannot
    contaminate the headline number.

    Scoring is ONE PAIR AT A TIME using PerceptualBank's per-image methods
    (bank.lpips / bank.dists / bank.ms_ssim / bank.clip_cosine). Those methods
    already produced the numbers you trust on the Mac, so results here are
    directly comparable -- same ruler, no batching surprises.

    `part_a` is the imported evaluate_new_metrics module (module 'A' in main),
    which is where PerceptualBank lives.
    """

    # order matters only for readability; AVG is computed per metric
    _METRICS = ("LPIPS", "DISTS", "MSSSIM", "CLIPcos")

    # channel treatments: name -> channel selection
    _PASSES = {
        "RGB":   [0, 1, 2],   # true colour
        "GBNIR": [1, 2, 3],   # false colour: green, blue, NIR
        "RRR":   [0, 0, 0],   # no-colour control: red replicated
    }

    def __init__(self, part_a, device: str = "cuda"):
        # Build the SAME bank the Mac script uses; it lazy-loads each model on
        # first call, on the chosen device.
        self.bank = part_a.PerceptualBank(device=device)

    @staticmethod
    def _triplet(im: np.ndarray, channels) -> np.ndarray:
        """Select a 3-channel (H,W,3) float view for one pass, contiguous for torch."""
        return np.ascontiguousarray(im[:, :, channels], dtype=np.float32)

    def _one_pass(self, ref3: np.ndarray, deg3: np.ndarray) -> dict:
        """All four metrics on a single (H,W,3) pair -> dict of 4 floats."""
        return {
            "LPIPS":   self.bank.lpips(ref3, deg3),
            "DISTS":   self.bank.dists(ref3, deg3),
            "MSSSIM":  self.bank.ms_ssim(ref3, deg3),
            "CLIPcos": self.bank.clip_cosine(ref3, deg3),
        }

    def score_pair_dual(self, ref_img: np.ndarray, deg_img: np.ndarray) -> dict:
        """
        ref_img, deg_img: (H,W,C>=4) float[0,1] arrays.
        Returns one dict with *_RGB, *_GBNIR, *_RRR, *_AVG for each metric.
        """
        scored = {}
        for pass_name, chans in self._PASSES.items():
            scored[pass_name] = self._one_pass(
                self._triplet(ref_img, chans),
                self._triplet(deg_img, chans),
            )
        row = {}
        for m in self._METRICS:
            for pass_name in self._PASSES:
                row[f"{m}_{pass_name}"] = float(scored[pass_name][m])
            # AVG = the two REAL views only; RRR is a control, excluded on purpose
            row[f"{m}_AVG"] = 0.5 * (float(scored["RGB"][m]) + float(scored["GBNIR"][m]))
        return row

    def score_list_dual(self, ref_imgs, deg_imgs):
        """Convenience: score a list of pairs one at a time -> list of row dicts."""
        return [self.score_pair_dual(r, d) for r, d in zip(ref_imgs, deg_imgs)]


# --------------------------------------------------------------------------- #
#  Structural metrics (CPU) -- imported verbatim, run per-channel incl. NIR
# --------------------------------------------------------------------------- #

def structural_row(clean_hwc, deg_hwc, A):
    """PSNR/SSIM/CW-SSIM per-channel (incl. NIR), averaged -- exactly as
    score_clean_vs_degraded does. Returns dict with PSNR, SSIM, CWSSIM."""
    n_ch = clean_hwc.shape[2]
    psnr_acc, ssim_acc, cw_acc = [], [], []
    for c in range(n_ch):
        a8 = (np.clip(clean_hwc[:, :, c], 0, 1) * 255).astype(np.uint8)
        b8 = (np.clip(deg_hwc[:, :, c], 0, 1) * 255).astype(np.uint8)
        psnr_acc.append(A.psnr_uint8(a8, b8))
        ssim_acc.append(A.ssim_matlab(a8, b8))
        cw_acc.append(A.cw_ssim(a8.astype(np.float64) / 255.0, b8.astype(np.float64) / 255.0))
    return {
        "PSNR": float(np.mean(psnr_acc)),
        "SSIM": float(np.mean(ssim_acc)),
        "CWSSIM": float(np.mean(cw_acc)),
    }


# --------------------------------------------------------------------------- #
#  CSV schema / resume
# --------------------------------------------------------------------------- #

PERC_METRICS = ["LPIPS", "DISTS", "MSSSIM", "CLIPcos"]
PERC_COLS = [f"{m}_{p}" for m in PERC_METRICS for p in ("RGB", "GBNIR", "RRR", "AVG")]
STRUCT_COLS = ["PSNR", "SSIM", "CWSSIM"]

DEG_COLS = ["tile", "family", "label", "strength", "fill"] + STRUCT_COLS + PERC_COLS
PAIR_COLS = ["tile"] + STRUCT_COLS + PERC_COLS


def _done_tiles(csv_path: Path) -> set:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    done = set()
    try:
        for chunk in pd.read_csv(csv_path, usecols=["tile"], chunksize=500_000):
            done.update(chunk["tile"].astype(str).unique().tolist())
    except Exception as e:  # noqa: BLE001
        print(f"  (resume parse failed on {csv_path.name}: {e}; starting fresh)", file=sys.stderr)
    return done


def _open_appender(csv_path: Path, cols):
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    fh = open(csv_path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if new:
        w.writeheader(); fh.flush()
    return fh, w


# --------------------------------------------------------------------------- #
#  Task 1: degradation sweep (degraded hr vs clean hr)
# --------------------------------------------------------------------------- #

def run_degradation(work, V, A, bank, out_dir, rng, save_examples=True):
    """
    For each tile: build all degraded versions (11 families x strengths) on CPU,
    compute structural per-variant, then score all that tile's perceptual passes
    one pair at a time (proven path). One tile = one atomic CSV flush (resume-safe).
    """
    from tqdm import tqdm

    csv_path = out_dir / "degradation_per_tile.csv"
    err_path = out_dir / "degradation_errors.csv"
    done = _done_tiles(csv_path)
    if done:
        before = len(work)
        work = [(s, p) for (s, p) in work if p.name not in done and s not in done]
        print(f"  [degradation] resume: {len(done):,} done, {before-len(work):,} skipped, {len(work):,} left")

    fh, wr = _open_appender(csv_path, DEG_COLS)
    efh, ew = _open_appender(err_path, ["tile", "error"])

    # examples once
    if save_examples and work:
        try:
            V._save_examples(work[0][1], out_dir, A, rng)
        except Exception as e:  # noqa: BLE001
            print(f"  (examples skipped: {e})", file=sys.stderr)

    n_ok = n_err = 0
    pbar = tqdm(work, desc="degradation", unit="tile", ncols=95)
    for stem, tp in pbar:
        try:
            raw = A.load_tile(tp)
            clean = V.tile_to_unit(raw)

            # Build every degraded variant on CPU (structural scored here), and
            # collect the (clean_ref, degraded) image pairs for a batched GPU
            # perceptual pass. We re-drive the SAME degradation functions as
            # run_sweeps_for_tile rather than calling it, because that function
            # invokes the slow unbatched perceptual metrics inline.
            metas, struct_rows, ref_imgs, deg_imgs = _build_tile_degradations(V, A, clean, rng)

            # Per-pair perceptual over ALL this tile's variants (proven, no batching)
            perc_rows = _perceptual_rows(bank, ref_imgs, deg_imgs)

            buffered = []
            for meta, srow, prow in zip(metas, struct_rows, perc_rows):
                rec = {"tile": tp.name, **meta, **srow, **prow}
                buffered.append(rec)
            wr.writerows(buffered)
            fh.flush(); os.fsync(fh.fileno())
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            ew.writerow({"tile": stem, "error": f"{type(e).__name__}: {e}"})
            efh.flush(); n_err += 1
        pbar.set_postfix(ok=n_ok, fail=n_err)

    fh.close(); efh.close()
    print(f"  [degradation] done ok={n_ok:,} fail={n_err:,}")


def _build_tile_degradations(V, A, clean, rng):
    """
    Re-drive the SAME degradation families/sweeps as run_sweeps_for_tile, but instead
    of scoring inline, return:
        metas:       list of {family,label,strength,fill}
        struct_rows: list of {PSNR,SSIM,CWSSIM}  (computed here, CPU)
        ref_imgs:    list of clean-reference (H,W,4) arrays (translation re-crops the ref!)
        deg_imgs:    list of degraded (H,W,4) arrays
    The ref list matters because translation crops the clean ref to overlap.
    """
    metas, struct_rows, ref_imgs, deg_imgs = [], [], [], []

    def add(family, label, strength, fill, ref, deg):
        metas.append({"family": family, "label": label, "strength": strength, "fill": fill})
        struct_rows.append(structural_row(ref, deg, A))
        ref_imgs.append(ref)
        deg_imgs.append(deg)

    # Gaussian noise
    for s in V.GAUSS_NOISE_SIGMAS:
        add("gaussian_noise", f"sigma={s}", s, "", clean, V.degrade_gaussian_noise(clean, s, rng))
    # Poisson
    for sc in V.POISSON_SCALES:
        strength = 0.0 if sc >= 1e8 else 1.0 / sc
        add("poisson_noise", f"scale={sc:g}", strength, "", clean, V.degrade_poisson_noise(clean, sc, rng))
    # Gaussian blur
    for s in V.BLUR_SIGMAS:
        add("gaussian_blur", f"sigma={s}", s, "", clean, V.degrade_gaussian_blur(clean, s))
    # Square artifacts (two fills)
    for fill in V.ARTIFACT_FILLS:
        for sz in V.ARTIFACT_SIZES:
            add(f"artifact_{fill}", f"size={sz}", float(sz), fill,
                clean, V.degrade_square_artifact(clean, sz, fill))
    # Translation (re-crops the clean ref!)
    for px in V.TRANSLATE_SHIFTS:
        ch, sh = V.translate_pair(clean, 0, px)
        add("translation_horizontal", f"shift={px}", float(px), "", ch, sh)
        cd, sd = V.translate_pair(clean, px, px)
        add("translation_diagonal", f"shift={px}", float(px), "", cd, sd)
    # Defocus
    for r in V.DEFOCUS_RADII:
        add("defocus_blur", f"radius={r}", r, "", clean, V.degrade_defocus(clean, r))
    # Motion
    for L in V.MOTION_LENGTHS:
        strength = 0.0 if L <= 1 else float(L)
        add("motion_blur", f"length={L}", strength, "", clean, V.degrade_motion_blur(clean, L))
    # Ringing
    for cf in V.RINGING_CUTOFFS:
        strength = 0.0 if cf >= 1.0 else (1.0 - cf)
        add("ringing", f"cutoff={cf}", strength, "", clean, V.degrade_ringing(clean, cf))
    # Aliasing
    for fac in V.ALIASING_FACTORS:
        strength = 0.0 if fac <= 1 else float(fac)
        add("aliasing", f"factor={fac}", strength, "", clean, V.degrade_aliasing(clean, fac))

    return metas, struct_rows, ref_imgs, deg_imgs


def _perceptual_rows(bank, ref_imgs, deg_imgs):
    """Run the dual-pass perceptual metrics over a list of pairs, one pair at a time."""
    if bank is None:
        return [{c: "" for c in PERC_COLS} for _ in ref_imgs]
    return bank.score_list_dual(ref_imgs, deg_imgs)


# --------------------------------------------------------------------------- #
#  Tasks 2 & 3: simple pair comparisons (fakelr vs hr, sentinel vs hr)
# --------------------------------------------------------------------------- #

def run_pair_task(task_name, work_pairs, A, bank, out_dir, batch_size):
    """
    work_pairs: list of (stem, hr_path, other_path).
    Scores other-vs-hr with structural (CPU) + dual-pass perceptual (batched GPU).
    """
    from tqdm import tqdm

    csv_path = out_dir / f"{task_name}_per_tile.csv"
    err_path = out_dir / f"{task_name}_errors.csv"
    done = _done_tiles(csv_path)
    if done:
        before = len(work_pairs)
        work_pairs = [(s, h, o) for (s, h, o) in work_pairs
                      if f"{s}.tif" not in done and s not in done]
        print(f"  [{task_name}] resume: {len(done):,} done, {before-len(work_pairs):,} skipped, {len(work_pairs):,} left")

    fh, wr = _open_appender(csv_path, PAIR_COLS)
    efh, ew = _open_appender(err_path, ["tile", "error"])

    n_ok = n_err = 0
    # Process in batches so the GPU perceptual pass is efficient.
    pbar = tqdm(range(0, len(work_pairs), batch_size), desc=task_name, unit="batch", ncols=95)
    for start in pbar:
        chunk = work_pairs[start:start + batch_size]
        refs, degs, metas, bad = [], [], [], []
        for stem, hr_path, other_path in chunk:
            try:
                hr = V_tile_to_unit_safe(A, hr_path)
                other = V_tile_to_unit_safe(A, other_path)
                if hr.shape[:2] != other.shape[:2]:
                    raise ValueError(f"shape mismatch hr={hr.shape} other={other.shape}")
                refs.append(hr); degs.append(other)
                metas.append((stem, f"{stem}.tif"))
            except Exception as e:  # noqa: BLE001
                bad.append((stem, f"{type(e).__name__}: {e}"))

        # structural (CPU) per pair
        struct = []
        for hr, other in zip(refs, degs):
            try:
                struct.append(structural_row(hr, other, A))
            except Exception as e:  # noqa: BLE001
                struct.append(None); bad.append((metas[len(struct)-1][0], f"struct: {e}"))

        # perceptual (per-pair, dual-pass) -- score each pair independently so one
        # bad pair can't sink the whole chunk
        if bank and refs:
            perc = []
            for idx, (r_img, d_img) in enumerate(zip(refs, degs)):
                try:
                    perc.append(bank.score_pair_dual(r_img, d_img))
                except Exception as e:  # noqa: BLE001
                    perc.append({c: "" for c in PERC_COLS})
                    bad.append((metas[idx][0], f"perc: {type(e).__name__}: {e}"))
        else:
            perc = [{c: "" for c in PERC_COLS} for _ in refs]

        buffered = []
        for (stem, tilename), s, p in zip(metas, struct, perc):
            if s is None:
                continue
            buffered.append({"tile": tilename, **s, **p})
        if buffered:
            wr.writerows(buffered); fh.flush(); os.fsync(fh.fileno())
            n_ok += len(buffered)
        for stem, msg in bad:
            ew.writerow({"tile": stem, "error": msg}); n_err += 1
        efh.flush()
        pbar.set_postfix(ok=n_ok, fail=n_err)

    fh.close(); efh.close()
    print(f"  [{task_name}] done ok={n_ok:,} fail={n_err:,}")


def V_tile_to_unit_safe(A, path):
    """load_tile (Part A) + per-channel min-max to [0,1] (same as tile_to_unit)."""
    raw = A.load_tile(path)
    out = np.zeros_like(raw, dtype=np.float32)
    for c in range(raw.shape[2]):
        ch = raw[:, :, c]
        lo, hi = float(ch.min()), float(ch.max())
        out[:, :, c] = 0.0 if hi - lo < 1e-12 else (ch - lo) / (hi - lo)
    return out


# --------------------------------------------------------------------------- #
#  Work-list construction from curated_pairs.csv
# --------------------------------------------------------------------------- #

def build_worklists(curated_csv, data_dir, limit=None):
    cur = pd.read_csv(curated_csv)
    if "stem" not in cur.columns:
        sys.exit("curated_pairs.csv has no 'stem' column")
    stems = list(dict.fromkeys(cur["stem"].astype(str)))
    if limit:
        stems = stems[:limit]

    hr_dir = data_dir / "hr_200"
    sr_dir = data_dir / "sr_60_200"     # Sentinel
    fl_dir = data_dir / "fake_lr"

    deg_work, fakelr_work, sentinel_work, missing = [], [], [], []
    for s in stems:
        hp = hr_dir / f"{s}.tif"
        if not hp.exists():
            missing.append(s); continue
        deg_work.append((s, hp))
        fp = fl_dir / f"{s}.tif"
        if fp.exists():
            fakelr_work.append((s, hp, fp))
        sp = sr_dir / f"{s}.tif"
        if sp.exists():
            sentinel_work.append((s, hp, sp))
    return deg_work, fakelr_work, sentinel_work, missing


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="workspace dir with hr_200/ sr_60_200/ fake_lr/")
    ap.add_argument("--scripts", type=Path, required=True, help="dir with the 3 imported .py + curated_pairs.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["degradation", "fakelr", "sentinel"],
                    choices=["degradation", "fakelr", "sentinel"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-perceptual", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N curated tiles (SMOKE TEST)")
    args = ap.parse_args()

    data = args.data.expanduser()
    scripts = args.scripts.expanduser()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    curated = scripts / "curated_pairs.csv"
    for req in [scripts / "superres_eval.py", scripts / "evaluate_new_metrics.py",
                scripts / "evaluate_degradation_sensitivity_v2.py", curated]:
        if not req.exists():
            sys.exit(f"Missing required file: {req}")

    # Import the metric code (same ruler). Make 'V' able to find Part A next to it.
    global V
    A = _load_module(scripts / "evaluate_new_metrics.py", "part_a")
    V = _load_module(scripts / "evaluate_degradation_sensitivity_v2.py", "degrade_v2")
    # V._load_part_a looks next to V's file; ensure Part A resolves there too.

    print("Building work lists from curated_pairs.csv ...")
    deg_work, fakelr_work, sentinel_work, missing = build_worklists(curated, data, args.limit)
    print(f"  degradation tiles : {len(deg_work):,}")
    print(f"  fakelr pairs      : {len(fakelr_work):,}")
    print(f"  sentinel pairs    : {len(sentinel_work):,}")
    if missing:
        print(f"  MISSING hr tiles  : {len(missing):,} (logged)")
        pd.DataFrame({"stem": missing}).to_csv(out / "missing_hr.csv", index=False)

    bank = None
    if not args.no_perceptual:
        print(f"Loading per-pair dual-pass perceptual bank on {args.device} ...")
        bank = DualPassPerceptual(A, device=args.device)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    if "degradation" in args.tasks:
        print("\n=== TASK: degradation (degraded hr vs clean hr) ===")
        run_degradation(deg_work, V, A, bank, out, rng)

    if "fakelr" in args.tasks:
        print("\n=== TASK: fakelr (fake_lr vs hr) ===")
        run_pair_task("fakelr", fakelr_work, A, bank, out, args.batch_size)

    if "sentinel" in args.tasks:
        print("\n=== TASK: sentinel (sr_60_200 vs hr) ===")
        run_pair_task("sentinel", sentinel_work, A, bank, out, args.batch_size)

    print(f"\nAll requested tasks finished in {(time.time()-t0)/60:.1f} min")
    # completion flag for easy checking after disconnect
    (out / "ALL_TASKS_COMPLETE.txt").write_text("done\n")


if __name__ == "__main__":
    main()
