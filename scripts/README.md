# RQ3 scripts: super-resolution training with three kinds of input

Three WaveCM diffusion models, identical except for the input they are trained on.
All three learn to produce PlanetScope.

| config | training input | status |
|---|---|---|
| `rq3_arm1_fakelr` | fake-LR (PlanetScope 192→60→192 bicubic) | ready |
| `rq3_arm2_s2like` | PlanetScope degraded like Sentinel-2 | **data not built yet** |
| `rq3_arm3_s2` | real Sentinel-2 | ready |

Budget per arm: 100,000 iterations, batch 64 (4 full A100-40GB × 16), about 12 h.
Test: 128 tiles, input = real Sentinel-2 for every arm.

## What is in this folder

| | |
|---|---|
| **Dataset (run on the Mac, already done)** | |
| `build_dataset.py` | picks train 5,000 / val 500 / test 128 (no shared AOIs), writes `Training Data/` + `manifest.csv` |
| `make_fakelr_testset.py` | the original fake-LR recipe script (reference; the recipe source) |
| `make_splits.py` | turns `manifest.csv` into `splits/*.csv` + fake-LR checksums |
| `splits/` | `train.csv`, `val.csv`, `val_monitor.csv` (64), `test.csv`, `fake_lr_checksums.csv` |
| **HAICORE** | |
| `env.sh` | all paths (workspace `srdata`, user `hgf_sma8852`); `source env.sh` before everything |
| `setup_env.sh` | one-time Python 3.12 + PyTorch install (Miniforge in `$HOME`) |
| `check_data.py` | checks every needed tile is on HAICORE and readable |
| `make_fake_lr.py` | makes our fake-LR on HAICORE and verifies it against the Mac checksums |
| `smoke_test.slurm` | 10–20 min end-to-end test (train → resume → evaluate) |
| `train.slurm` | trains one arm (4 full A100s, 48 h limit, resumes by itself) |
| `eval.slurm` | super-resolves the 128 test tiles with an arm's best checkpoint |
| `make_configs.py` → `config/` | the three training configs, generated from one template |
| **Model code** (WaveCM teacher, from github.com/lucpaul/SR3_Wavelet_Superres4Sentinel2 @ 59e295d) | |
| `sr.py`, `core/`, `data/`, `model/`, `requirements.txt`, `LICENSE` | the training code; what we changed and why is in `CHANGES.md` |

## Run on HAICORE@KIT

### 0. Upload (from the Mac)
```bash
rsync -av --exclude build_dataset.py --exclude make_fakelr_testset.py --exclude .DS_Store ~/Desktop/neural-posterior-estimation/scripts/ \
  hgf_sma8852@haicore.scc.kit.edu:~/rq3/
rsync -av ~/Desktop/Super_resolution_project/Test/hr_200 ~/Desktop/Super_resolution_project/Test/sr_60_200 \
  hgf_sma8852@haicore.scc.kit.edu:/hkfs/work/workspace_haic/scratch/hgf_sma8852-srdata/Test/
```

### 1. One-time setup (login node, ~10 min)
```bash
cd ~/rq3
bash setup_env.sh 2>&1 | tail -5        # ends with: torch 2.9.1+cu128 ... Environment ready
```

### 2. Check data, make fake-LR
```bash
cd ~/rq3 && source env.sh && source ~/miniforge3/bin/activate $RQ3_VENV
python check_data.py                    # ALL GOOD
python make_fake_lr.py                  # ~5–10 min; 0 DIFFERENT
python check_data.py --with-fakelr      # ALL GOOD
```
If `make_fake_lr.py` reports tiles as DIFFERENT, don't train arm 1. Upload the Mac tiles instead
(`Training Data/{train,val,test}/fake_lr/*` → `$RQ3_DATA/rq3_fake_lr/`).

### 3. Smoke test (1 GPU, 10–20 min)
```bash
sbatch smoke_test.slurm
tail -f logs/rq3_smoke_*.out            # must end with: SMOKE TEST PASSED
```

### 4. Train
```bash
sbatch train.slurm rq3_arm1_fakelr
sbatch train.slurm rq3_arm3_s2
squeue -u $USER
grep "Validation # PSNR" $RQ3_RUNS/rq3_arm1_fakelr/logs/train.log
```
If a job ever stops early, submit it again; it resumes from the newest checkpoint.
The best checkpoint (validation PSNR) is `$RQ3_RUNS/<arm>/checkpoint/<arm>_best_gen.pth`.

### 5. Evaluate on the 128 test tiles
```bash
sbatch eval.slurm rq3_arm1_fakelr          # input = real Sentinel-2
sbatch eval.slurm rq3_arm3_s2
sbatch eval.slurm rq3_arm1_fakelr own      # optional: arm 1 on its own input type
```
Output: `$RQ3_RUNS/<arm>/eval_<input>/<stem>.tif` (float32, H,W,C, reflectance) + `quick_psnr.csv`.

## HAICORE facts (checked 20 Oct 2026)
- Partition `normal`: max 3 days per job, 1 node.
- **Always request `gpu:full:N`.** Plain `gpu:N` can hand out MIG slices (A100 split into 20 GB / 10 GB pieces).
- System Python is 3.9 and there is no Python module, hence Miniforge.
- Workspace `srdata` expires **19 Dec 2026** (1 extension left: `ws_extend srdata 60`).
- **HAICORE@KIT is offline 23 Nov – 4 Dec 2026**, so runs must finish before then.

## If something goes wrong
- `Unset environment variable in config value` → `source env.sh` was skipped.
- `Missing N stems in ...` → a folder doesn't match the split lists; run `python check_data.py`.
- Fewer than 4 GPUs: keep the **total** batch at 64 (e.g. 2 GPUs → `batch_size` 32 in `make_configs.py`, re-run it), so the arms stay comparable.
