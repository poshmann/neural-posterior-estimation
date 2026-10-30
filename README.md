# Simulation-based inference of Sentinel-2 degradations for super-resolution

MSc thesis (Data Science, University of Potsdam), Poshan Pokhanal.
Supervisors: Dr. Silvia Makowski (University of Potsdam), Dr. Hui Tang (GFZ Potsdam).

Super-resolution models for Sentinel-2 are usually trained on "fake" low-resolution images made by
downsampling PlanetScope. Real Sentinel-2 differs from that in ways the fake images miss. This thesis
measures *how* it differs and whether training on realistic degradations helps.

## Research questions

| | Question | Approach |
|---|---|---|
| **RQ1** | Can the degradation of a Sentinel-2 tile, θ = (blur σ_b, shift r, noise σ_n), be inferred from a 7-metric image-quality fingerprint, with calibrated uncertainty? | Neural posterior estimation (NPE, `sbi`): a forward simulator degrades PlanetScope tiles, an ensemble of normalising flows learns p(θ \| fingerprint); validated with SBC, coverage and a rejection-ABC baseline. |
| **RQ2** | What does the posterior say about real Sentinel-2 / PlanetScope pairs? Which parameters are identifiable, which confounded, and is the forward model adequate? | The NPE applied to 3,000 real pairs; independent shift measurement (phase correlation); posterior predictive checks. |
| **RQ3** | Does super-resolution trained on Sentinel-2-like degradations do better on real Sentinel-2? | The WaveCM diffusion SR model trained four ways and evaluated on 128 real test pairs (below). |

RQ3 training arms (identical model, data split and HR targets; only the training input differs):

| Arm | Training input |
|---|---|
| 1 `rq3_arm1_fakelr` | fake-LR (PlanetScope 192 → 60 → 192, bicubic, anti-aliased) |
| 2 `rq3_arm2_s2like` | PlanetScope degraded with θ drawn from the NPE posterior of the tile's own real Sentinel-2 pair, plus quantile-mapped radiometry |
| 2b `rq3_arm2b_mshift` | as arm 2, but with the shift magnitude measured directly on the real pair |
| 3 `rq3_arm3_s2` | real Sentinel-2 |

## Repository layout

```
scripts/                 all code (pipeline, commands and HAICORE usage in scripts/README.md)
  simulator.py           forward model: blur -> sub-pixel shift -> bicubic 60x60 -> noise -> bicubic back
  simulate.py/.slurm     simulations and fingerprints (train, val, theta=0 baseline, real pairs)
  npe.py                 NPE: train (grid + ensemble), eval, apply to real fingerprints
  scene_stats.py         scene-contrast inputs for the NPE
  prepare_real.py        real Sentinel-2 / fake-LR fingerprints of the 3,000 paper pairs
  rq2_analysis.py        RQ2 summaries (incl. split by tile family)
  shift_check.py         measured S2-vs-PlanetScope shift (phase correlation) vs the NPE shift
  ppc.py/.slurm          posterior predictive check
  radiometry_fits.py     S2-vs-PlanetScope brightness relation (for arm 2)
  build_arm2.py          arm 2 / 2b training inputs
  build_dataset.py, make_splits.py, make_fake_lr.py, check_data.py   RQ3 dataset (leakage-free split)
  sr.py, core/, data/, model/, config/, make_configs.py              WaveCM diffusion SR (see CHANGES.md)
  train.slurm, eval.slurm, evaluate_rq3.py/.slurm                    RQ3 training and evaluation
npe_dev/                 early development reports (structural metrics only)
Thesis_working_notes_2026-10-30.docx   detailed notes: data, methods, settings, results, commands
handoff-2026-10-30.md    current status, running jobs, next steps
```

## Data and results (not in this repository)

- **Imagery:** WaveCM dataset (Sentinel-2 `COPERNICUS/S2_SR_HARMONIZED`, PlanetScope `ortho_analytic_4b_sr`),
  600 m tiles (60 Sentinel-2 px / 200 PlanetScope px), 4 bands. The RQ3 set (5,000 train, 500 val,
  128 test tiles, leakage-free by AOI/scene) is built with `scripts/build_dataset.py`.
- **Models, posteriors, simulations and SR outputs** are kept locally (`npe_full/`, `results_haicore/`)
  and on the HAICORE cluster (KIT); they are excluded via `.gitignore` because of their size.

## Status (30 October 2026)

- **NPE v4** (shift prior up to 2 Sentinel-2 px, 100k simulations): calibrated on held-out simulations
  (95 % coverage 0.94 for all three parameters).
- **RQ2:** real Sentinel-2 medians σ_b 1.75 PlanetScope px, r 2.29 px, σ_n 0.0041 reflectance; two tile
  families (AOI_* and tile_*) with different processing. Large shifts are real per-scene georeferencing
  offsets, but the NPE shift is ~0.7 px too large on average, and about half of the real pairs are
  perceptually more different than any blur/shift/noise setting (forward-model misspecification).
- **RQ3:** 100k-iteration pilot done for arms 1 and 3; the 300k-iteration main runs (4 arms × 2 seeds)
  are running on HAICORE.

Details and all numbers: `Thesis_working_notes_2026-10-30.docx`.

## Software

Python 3.12, PyTorch 2.9 (HAICORE, CUDA 12.8), `sbi` for NPE. Pinned versions: `scripts/requirements.txt`.
The SR model is based on [SR3_Wavelet_Superres4Sentinel2](https://github.com/lucpaul/SR3_Wavelet_Superres4Sentinel2)
(WaveCM teacher); our changes are listed in `scripts/CHANGES.md`.
