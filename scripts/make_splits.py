#!/usr/bin/env python3
"""Run ONCE on the Mac: turn manifest.csv into the split lists the training reads, and
record checksums of the Mac fake-LR tiles so HAICORE can verify its regenerated copies.

    python make_splits.py "~/Desktop/neural-posterior-estimation/Training Data"

Writes splits/:
  train.csv (5,000)  val.csv (500)  test.csv (128)    columns: stem, group
  val_monitor.csv (64)  fixed subset of val, used during training to pick the best checkpoint
                        (2000-step sampling of all 500 every 10k iterations would cost ~5 h/arm)
  fake_lr_checksums.csv  stem, md5, mean, std of every Mac fake-LR tile
"""
import csv, hashlib, os, sys
import numpy as np
import tifffile

SEED, N_MONITOR = 20261020, 64


def main(root):
    root = os.path.expanduser(root)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(root, "manifest.csv"))))
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    for s, rs in by.items():
        with open(os.path.join(out, f"{s}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["stem", "group"])
            w.writerows([r["stem"], r["group"]] for r in rs)
        print(f"{s}.csv: {len(rs)}")

    # monitor subset: round-robin over val groups so every group is represented
    rng = np.random.default_rng(SEED)
    groups = {}
    for r in by["val"]:
        groups.setdefault(r["group"], []).append(r)
    order = [groups[g] for g in sorted(groups)]
    order = [order[i] for i in rng.permutation(len(order))]
    pools = [[g[i] for i in rng.permutation(len(g))] for g in order]
    pick = []
    while len(pick) < N_MONITOR:
        for p in pools:
            if p and len(pick) < N_MONITOR:
                pick.append(p.pop())
    with open(os.path.join(out, "val_monitor.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["stem", "group"])
        w.writerows([r["stem"], r["group"]] for r in sorted(pick, key=lambda r: r["stem"]))
    print(f"val_monitor.csv: {len(pick)} tiles from {len({r['group'] for r in pick})} groups")

    with open(os.path.join(out, "fake_lr_checksums.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["stem", "md5", "mean", "std"])
        for r in rows:
            a = tifffile.imread(os.path.join(root, r["split"], "fake_lr", r["stem"] + ".tif"))
            a = np.ascontiguousarray(a, dtype=np.float32)
            w.writerow([r["stem"], hashlib.md5(a.tobytes()).hexdigest(),
                        repr(float(a.mean(dtype=np.float64))), repr(float(a.std(dtype=np.float64)))])
    print("fake_lr_checksums.csv:", len(rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/Desktop/neural-posterior-estimation/Training Data")
