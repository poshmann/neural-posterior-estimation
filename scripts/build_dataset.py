#!/usr/bin/env python3
"""
Build the thesis RQ3 dataset (train 5,000 / val 500 / test 128) from the WaveCM data.

Arms share the SAME HR tiles; only the LR input differs:
  hr/         PlanetScope target                 (copied, cropped to 192)
  lr/         real Sentinel-2 (sr_60_200)        (cropped to 192, stored H,W,C) -> arm 3
  fake_lr/    PlanetScope 192 -> 60 -> 192 bicubic, skimage anti_aliasing=True -> arm 1
  (arm 2, PlanetScope degraded like Sentinel-2, is generated later from hr/)

Split rules
  * test  = the existing 128 WaveCM test tiles (Test/hr_200, Test/sr_60_200)
  * every AOI / scene that has a test tile is removed from train/val (no spatial leakage)
  * train and val come from disjoint AOIs/scenes
  * per-group cap ("water-filling"): smallest cap c with sum_g min(n_g, c) >= target,
    so no single scene dominates
  * tiles with NaN/inf, a constant band or >1% exact zeros (HR or S2) are skipped
  * 200x200 tiles are cropped top-left to 192 (as in the original eval pipeline)
  * every output file is (H,W,C), 4 bands, one page -- same layout as the PlanetScope source

Stages (idempotent, re-run safely):
  python build_rq3_dataset.py select     # writes manifest.csv
  python build_rq3_dataset.py build      # writes tiles listed in manifest (skips existing)
  python build_rq3_dataset.py verify
"""
import os, sys, csv, time, shutil, collections
import numpy as np, tifffile
from skimage.transform import resize

SRC   = os.path.expanduser(os.environ.get("SRC", "~/Desktop/Super_resolution_project"))
OUT   = os.path.expanduser(os.environ.get("OUT", "~/Desktop/neural-posterior-estimation/Training Data"))
SEED  = 20261020
N_TRAIN, N_VAL = 5000, 500
VAL_GROUP_FRAC = 0.10
SIZE, DOWN = 192, 60
TRAIN_HR, TRAIN_S2 = f"{SRC}/Training Data/hr_200", f"{SRC}/Training Data/sr_60_200"
TEST_HR,  TEST_S2  = f"{SRC}/Test/hr_200",          f"{SRC}/Test/sr_60_200"
MANIFEST = f"{OUT}/manifest.csv"

def tifs(d): return sorted(f for f in os.listdir(d) if f.endswith(".tif") and not f.startswith("."))
def group(f):
    s = f[:-4].split("_")
    return f"AOI_{s[1]}" if s[0] == "AOI" else f"scene_{s[2]}"

def hwc(a):  # -> (H,W,C)
    return np.moveaxis(a, 0, -1) if a.shape[0] == 4 and a.shape[-1] != 4 else a
def crop(a, chw):
    return a[:, :SIZE, :SIZE] if chw else a[:SIZE, :SIZE, :]
def is_chw(a): return a.shape[0] == 4 and a.shape[-1] != 4

def ok(a):
    a = hwc(a).astype(np.float32)
    if a.shape[0] < SIZE or a.shape[1] < SIZE or a.shape[2] != 4: return False
    if not np.isfinite(a).all(): return False
    if (a.reshape(-1, 4).std(0) == 0).any(): return False
    return (a == 0).mean() <= 0.01

def cap_for(sizes, target):
    lo, hi = 1, max(sizes)
    while lo < hi:
        c = (lo + hi) // 2
        if sum(min(n, c) for n in sizes) >= target: hi = c
        else: lo = c + 1
    return lo

def draw(groups, target, rng, check):
    """Water-filling capped draw. Walk each group's shuffled list, keeping tiles that
    pass the quality check, until the group holds `cap` good tiles; raise the cap until
    the target is reached, then randomly trim the surplus from the fullest groups."""
    order = {g: [fs[i] for i in rng.permutation(len(fs))] for g, fs in groups.items()}
    pos = {g: 0 for g in groups}; kept = {g: [] for g in groups}; rejected = 0
    c = cap_for([len(v) for v in groups.values()], target)
    while True:
        for g in groups:
            while len(kept[g]) < c and pos[g] < len(order[g]):
                f = order[g][pos[g]]; pos[g] += 1
                if check(f): kept[g].append(f)
                else: rejected += 1
        total = sum(map(len, kept.values()))
        if total >= target: break
        if all(pos[g] >= len(order[g]) for g in groups): raise SystemExit("ran out of tiles")
        c += 1
    while total > target:          # trim from groups currently at the cap
        full = [g for g in groups if len(kept[g]) == max(map(len, kept.values()))]
        g = full[rng.integers(len(full))]; kept[g].pop(); total -= 1
    return [(g, f) for g in groups for f in kept[g]], rejected

def select():
    rng = np.random.default_rng(SEED)
    pool = sorted(set(tifs(TRAIN_HR)) & set(tifs(TRAIN_S2)))
    test = sorted(set(tifs(TEST_HR)) & set(tifs(TEST_S2)))
    tgroups = set(map(group, test))
    clean = [f for f in pool if group(f) not in tgroups]
    by = collections.defaultdict(list)
    for f in clean: by[group(f)].append(f)
    gl = sorted(by); gl = [gl[i] for i in rng.permutation(len(gl))]
    nval = max(1, round(VAL_GROUP_FRAC * len(gl)))
    vg, tg = set(gl[:nval]), set(gl[nval:])

    def check(f):
        if f in check.cache: return check.cache[f]
        try: r = ok(tifffile.imread(f"{TRAIN_HR}/{f}")) and ok(tifffile.imread(f"{TRAIN_S2}/{f}"))
        except Exception: r = False      # empty / unreadable file
        check.cache[f] = r; return r
    check.cache = {}

    val, rv = draw({g: by[g] for g in vg}, N_VAL, rng, check)
    train, rt = draw({g: by[g] for g in tg}, N_TRAIN, rng, check)
    os.makedirs(OUT, exist_ok=True)
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["split", "stem", "group", "hr_src", "s2_src"])  # paths relative to SRC
        for split, rows in (("train", train), ("val", val)):
            for g, f in sorted(rows, key=lambda r: r[1]):
                w.writerow([split, f[:-4], g, f"Training Data/hr_200/{f}", f"Training Data/sr_60_200/{f}"])
        for f in test:
            w.writerow(["test", f[:-4], group(f), f"Test/hr_200/{f}", f"Test/sr_60_200/{f}"])
    tc = collections.Counter(g for g, _ in train); vc = collections.Counter(g for g, _ in val)
    print(f"pool {len(pool)} | test groups removed {len(tgroups)} -> clean {len(clean)} tiles / {len(by)} groups")
    print(f"val groups {len(vg)}, train groups {len(tg)}")
    print(f"train {len(train)} from {len(tc)} groups, max/group {max(tc.values())}, rejected {rt}")
    print(f"val   {len(val)} from {len(vc)} groups, max/group {max(vc.values())}, rejected {rv}")
    print(f"test  {len(test)}")

def fakelr(hr_hwc):
    def b(a, n): return resize(a, (n, n, a.shape[2]), order=3, mode="reflect",
                               anti_aliasing=True, preserve_range=True).astype(np.float32)
    return b(b(hr_hwc, DOWN), SIZE)

def save(path, arr):
    """Write like the WaveCM PlanetScope tiles: ONE page, 4 samples per pixel (H,W,C,
    planar contig), photometric RGB + 1 extra sample, uncompressed float32.
    Sentinel-2 is converted from its source (C,H,W) layout to the same (H,W,C)."""
    tifffile.imwrite(path, np.ascontiguousarray(arr, dtype=np.float32), photometric="rgb",
                     planarconfig="contig", extrasamples=(2,))

def good(path):
    try:
        with tifffile.TiffFile(path) as t:
            p = t.pages[0]
            return (len(t.pages) == 1 and p.samplesperpixel == 4 and p.planarconfig == 1
                    and p.imagewidth == SIZE and p.imagelength == SIZE)
    except Exception:
        return False

def build(budget=150):
    t0 = time.time(); done = skipped = 0
    rows = list(csv.DictReader(open(MANIFEST)))
    for r in rows:
        d = f"{OUT}/{r['split']}"
        outs = {k: f"{d}/{k}/{r['stem']}.tif" for k in ("hr", "lr", "fake_lr")}
        if all(good(p) for p in outs.values()): skipped += 1; continue
        if time.time() - t0 > budget: break
        for k in outs: os.makedirs(os.path.dirname(outs[k]), exist_ok=True)
        srcs = {"hr": f"{SRC}/{r['hr_src']}", "lr": f"{SRC}/{r['s2_src']}"}
        arrs = {}
        for k, p in srcs.items():
            a = tifffile.imread(p); chw = is_chw(a)
            if not chw and hwc(a).shape[:2] == (SIZE, SIZE):
                shutil.copyfile(p, outs[k])           # untouched HR tile: byte-identical copy
            else:                                     # crop 200 -> 192 and/or S2 (C,H,W) -> (H,W,C)
                save(outs[k], hwc(crop(a, chw)))
            arrs[k] = hwc(crop(a, chw)).astype(np.float32)
        save(outs["fake_lr"], fakelr(arrs["hr"]))
        done += 1
    left = len(rows) - skipped - done
    print(f"built {done}, already there {skipped}, remaining {left}")

def verify():
    rows = list(csv.DictReader(open(MANIFEST)))
    c = collections.Counter(r["split"] for r in rows); print("manifest:", dict(c))
    g = {s: set(r["group"] for r in rows if r["split"] == s) for s in c}
    print("group overlap train/val:", len(g["train"] & g["val"]),
          " train/test:", len(g["train"] & g["test"]), " val/test:", len(g["val"] & g["test"]))
    bad = 0
    for r in rows:
        for k in ("hr", "lr", "fake_lr"):
            p = f"{OUT}/{r['split']}/{k}/{r['stem']}.tif"
            if not os.path.exists(p): bad += 1; continue
            if not good(p): bad += 1; continue
            a = hwc(tifffile.imread(p))
            if a.shape != (SIZE, SIZE, 4) or not np.isfinite(a).all(): bad += 1
    print("missing or malformed files:", bad)

if __name__ == "__main__":
    {"select": select, "build": build, "verify": verify}[sys.argv[1]]()
