#!/usr/bin/env python3
"""Write the three RQ3 training configs from ONE template, so the arms differ only in
their input folder and name. Re-run after changing anything here.

    python make_configs.py          -> config/rq3_arm{1,2,3}_*.json

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
        "save_checkpoint_freq": 5000,
        "keep_checkpoints": 3,
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
    for name, sr_dir in ARMS.items():
        cfg = copy.deepcopy(TEMPLATE)
        cfg["name"] = name
        cfg["path"]["experiments_root"] = "$RQ3_RUNS/" + name
        cfg["datasets"]["train"]["sr_dir"] = sr_dir
        cfg["datasets"]["val"]["sr_dir"] = sr_dir
        out = os.path.join(CONFIG_DIR, name + ".json")
        with open(out, "w") as fh:
            json.dump(cfg, fh, indent=4)
        print("wrote", os.path.relpath(out))


if __name__ == "__main__":
    main()
