#!/usr/bin/env python3
"""Write the three RQ3 training configs from ONE template, so the arms differ only in
their input folder and name. Re-run after changing anything here.

    python make_configs.py          -> config/rq3_arm{1,2,3}_*.json        (100k iterations)
                                       config/rq3_arm{1,2,3}_*_300k.json   (300k iterations)

Paths use environment variables, set in env.sh on HAICORE:
    RQ3_DATA    folder holding hr_200/, sr_60_200/ (the uploaded WaveCM training data)
                and rq3_fake_lr/ (made by make_fake_lr.py)
    RQ3_TEST    folder holding the 128 test tiles: hr_200/, sr_60_200/
    RQ3_SPLITS  folder with train.csv, val.csv, val_monitor.csv, test.csv
    RQ3_RUNS    where runs (checkpoints, logs, results) are written
"""
import copy, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "config")

# arm -> folder with the model INPUT for train/val (relative to $RQ3_DATA)
ARMS = {
    "rq3_arm1_fakelr": "rq3_fake_lr",   # PlanetScope 192->60->192 bicubic (anti-aliased)
    "rq3_arm2_s2like": "rq3_s2like",    # PlanetScope degraded like Sentinel-2 (NOT BUILT YET)
    "rq3_arm3_s2":     "sr_60_200",     # real Sentinel-2 (upsampled to the PlanetScope grid)
    "rq3_arm2b_mshift": "rq3_s2like_mshift",  # arm 2 with the MEASURED shift magnitude (build_arm2.py --shift measured)
}

# budget suffix -> iterations. The cosine LR schedule follows n_iter (T_max = n_iter, model/model.py).
# 29 Oct 2026: arm 3 was still improving at 100k (val PSNR +5 dB from 60k to 90k) -> rerun at 300k.
# The 100k runs are kept untouched in their own folders (budget sensitivity).
# 29 Oct 2026 (later): the WaveCM paper (von Chamier et al., sec. 5.4) trained the DiWa teacher with
# lr 1e-4 (cosine to 1e-7); 4e-5 in the upstream config is the paper's consistency-STUDENT lr.
# The 300k runs use the paper's teacher lr. (100k runs: 4e-5, kept as they were run.)
# Seed 2 (29 Oct 2026): identical to _300k except the seed (initial weights, data order, diffusion noise).
# 30 Oct 2026: all four 300k runs (fp16 AMP, lr 1e-4) crashed after 400-3,700 iterations: single updates
# pushed activations beyond the fp16 range (inf loss on all GPUs at once). 300k runs therefore use bf16
# autocast (float32 range, no loss scaling) and a 1,000-iteration linear lr warm-up; model.py also skips
# invalid steps safely now. The 100k configs are unchanged (fp16, no warm-up).
BUDGETS = {          # suffix -> (iterations, Adam lr, seed, amp dtype, warm-up iterations)
    "":            (100000, 4e-5, 42, None, None),
    "_300k":       (300000, 1e-4, 42, "bf16", 1000),
    "_300k_seed2": (300000, 1e-4, 43, "bf16", 1000),
}

TEMPLATE = {
    "name": None,
    "phase": "train",
    "gpu_ids": [0, 1, 2, 3],
    "path": {
        "experiments_root": None,      # fixed folder -> chained jobs resume in place
        "log": "logs",
        "tb_logger": "tb",
        "results": "results",
        "checkpoint": "checkpoint",
        "resume_state": "auto",        # newest checkpoint in this run, else fresh start
    },
    "datasets": {
        "train": {
            "name": "Sentinel2PlanetScope", "dataroot": "$RQ3_DATA",
            "hr_dir": "hr_200", "sr_dir": None,
            "index_csv": "$RQ3_SPLITS/train.csv",
            "mode": "HR", "datatype": "img", "l_resolution": 60, "r_resolution": 200,
            "batch_size": 16,          # per GPU; 4 GPUs -> 64 total, as in the original run
            "num_workers": 8, "use_shuffle": True, "data_len": -1,
        },
        "val": {                        # monitored during training; picks the best checkpoint
            "name": "Sentinel2PlanetScope", "dataroot": "$RQ3_DATA",
            "hr_dir": "hr_200", "sr_dir": None,
            "index_csv": "$RQ3_SPLITS/val_monitor.csv",
            "mode": "HR", "datatype": "img", "l_resolution": 60, "r_resolution": 200,
            "batch_size": 8, "num_workers": 4, "use_shuffle": False, "data_len": -1,
        },
        "test": {                       # only used with -p val (evaluation)
            "name": "Sentinel2PlanetScope", "dataroot": "$RQ3_TEST",
            "hr_dir": "hr_200", "sr_dir": "$EVAL_INPUT_DIR",
            "index_csv": "$RQ3_SPLITS/test.csv",
            "mode": "HR", "datatype": "img", "l_resolution": 60, "r_resolution": 200,
            "batch_size": 8, "num_workers": 4, "use_shuffle": False, "data_len": -1,
        },
    },
    "model": {   # unchanged from the original WaveCM teacher (config/sr_sr3_wave.json)
        "which_model_G": "wave",
        "finetune_norm": False,
        "unet": {
            "in_channel": 32, "out_channel": 16, "inner_channel": 128, "norm_groups": 16,
            "channel_multiplier": [1, 2, 4, 4, 8, 8, 16], "attn_res": [16],
            "res_blocks": 3, "dropout": 0,
        },
        "beta_schedule": {
            "train": {"schedule": "linear", "n_timestep": 2000, "linear_start": 1e-6, "linear_end": 1e-2},
            "val":   {"schedule": "linear", "n_timestep": 2000, "linear_start": 1e-6, "linear_end": 1e-2},
        },
        "diffusion": {"image_size": 192, "channels": 4, "conditional": True},
    },
    "train": {
        "n_iter": 100000,              # fixed budget, identical for all arms (~1,280 epochs of 5,000 tiles)
        "val_freq": 10000,
        "save_checkpoint_freq": 10000,   # one checkpoint set is ~15 GB
        "keep_checkpoints": 1,          # newest only (enough to resume)
        "print_freq": 100,
        "seed": 42,                    # identical initial weights in every arm
        "use_amp": True,
        "val_clip": False,             # CLIP needs a model download; compute nodes are offline
        "optimizer": {"type": "adam", "lr": 4e-5},   # code uses Adam (no weight decay), as upstream
    },
    "eval": {"color_fix": False},      # no AdaIN colour matching to the input at test time
    "wandb": {"project": "rq3"},
}


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    for arm, sr_dir in ARMS.items():
        for suffix, (n_iter, lr, seed, amp_dtype, warmup) in BUDGETS.items():
            name = arm + suffix
            cfg = copy.deepcopy(TEMPLATE)
            cfg["name"] = name
            cfg["path"]["experiments_root"] = "$RQ3_RUNS/" + name
            cfg["datasets"]["train"]["sr_dir"] = sr_dir
            cfg["datasets"]["val"]["sr_dir"] = sr_dir
            cfg["train"]["n_iter"] = n_iter
            cfg["train"]["optimizer"]["lr"] = lr
            cfg["train"]["seed"] = seed
            if amp_dtype:
                cfg["train"]["amp_dtype"] = amp_dtype
            if warmup:
                cfg["train"]["warmup_iters"] = warmup
            out = os.path.join(CONFIG_DIR, name + ".json")
            with open(out, "w") as fh:
                json.dump(cfg, fh, indent=4)
            print("wrote", os.path.relpath(out))


if __name__ == "__main__":
    main()
