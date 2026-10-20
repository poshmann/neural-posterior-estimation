# RQ3 changes to the WaveCM training code

Base: `github.com/lucpaul/SR3_Wavelet_Superres4Sentinel2` @ `59e295d` (8 Jun 2026), the
code that trained the WaveCM diffusion teacher. Only the files needed for the wavelet
teacher are kept (removed: `infer.py` (broken upstream), `sample.py`, the old SLURM scripts and
configs, `model/ddpm_modules`, `model/sr3_modules`, the CelebA example data).

**The model, diffusion and loss are NOT changed** (U-Net, DWSR, Haar wavelets, T = 2000,
linear β 1e-6 → 1e-2, weighted wavelet L1 loss 1.5/2/2/1, Adam lr 4e-5, AMP, batch 16 × 4 GPUs).
What changed is the plumbing around it:

| File | Change | Why |
|---|---|---|
| `data/LRHR_dataset.py` | Input and HR tiles are converted to (H,W,C) by finding the 4-band axis | Upstream always transposed the input from (C,H,W): right for old Sentinel-2 files, **wrong for fake-LR** and our re-saved tiles |
| | `hr_dir` / `sr_dir` settable per config | Arms differ only in their input folder |
| | HR ↔ input matched by stem, direct file lookup | Safer, and fast on 115k-file folders |
| `data/__init__.py` | Pass the new options; allow 0 workers | — |
| `core/logger.py` | `$VARS` in config paths; fixed run folder (`path.experiments_root`); `resume_state: "auto"` = newest complete checkpoint | Chained 24 h SLURM jobs resume in place |
| | Logs appended, not overwritten | Upstream wiped `train.log` on every restart |
| | `-debug` stops after 12 iterations | Smoke test |
| `model/model.py` | Cosine LR decays over `train.n_iter` (upstream hard-coded `T_max=750000`) | With a 100k budget the LR would otherwise barely decay |
| | AMP switchable (`train.use_amp`), CPU-safe | — |
| | Keep only the newest `train.keep_checkpoints` periodic checkpoints | Disk |
| `sr.py` | Seed set **before** the model is built (`train.seed`) | Identical initial weights in every arm |
| | Per-GPU seeds after init, shifted by the resume step | Different noise per GPU; resumed runs don't replay the same noise |
| | Best val PSNR stored in `checkpoint/best_val.json` | Upstream reset it to 0 on restart, so any later checkpoint became "best" |
| | CLIP in validation optional (`train.val_clip`, off) | Compute nodes have no internet for the CLIP download |
| | Evaluation (`-p val`) uses the config's `test` block; writes each SR tile as float32 (H,W,C) reflectance + `quick_psnr.csv`; logs to `eval.log` | Clean outputs for the thesis evaluation |
| | Colour matching (AdaIN) to the input **off** unless `eval.color_fix: true` | Upstream copied the input's colours onto the output, which biases a comparison between input types |
| | `--resume_state`, `--results_dir` options | Evaluation jobs |
| `requirements.txt` | + `opencv-python-headless` | `core/metrics.py` imports `cv2` but upstream didn't list it |
| `model/networks.py` | Only the wavelet model can be selected (other variants removed) | — |
| new files | `make_configs.py`, `splits/`, `check_data.py`, `make_fake_lr.py`, `env.sh`, `setup_env.sh`, `*.slurm` | — |

Not changed, but worth knowing:
- The original `config/sr_sr3_wave.json` (not kept here) said `adamw` with weight decay, but the code always used plain **Adam** (no weight decay). We keep the code's behaviour.
- `ema_scheduler` in the config is unused by the code (no EMA in the teacher).
- The original validation used the 128 **test** tiles. We validate on a 64-tile subset of our own val split.
- Validation PSNR inside `sr.py` min-max normalises each batch. It is only used to pick the checkpoint. Thesis metrics come from a separate evaluation.
