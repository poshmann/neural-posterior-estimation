#!/usr/bin/env python3
"""Download the perceptual-metric weights ONCE on a login node (compute nodes may be
offline): LPIPS (AlexNet), DISTS (VGG16 + DISTS weights), CLIP ViT-B-32 laion2b_s34b_b79k.
Then score one random pair to prove everything loads.

    source env.sh && source ~/miniforge3/bin/activate $RQ3_VENV && python prefetch_weights.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator as S  # noqa: E402

fp = S.Fingerprinter(device="cpu", perceptual=True)
rng = np.random.default_rng(0)
a = rng.random((192, 192, 4)).astype(np.float32)
b = np.clip(a + 0.05 * rng.standard_normal(a.shape).astype(np.float32), 0, 1)
row = fp(a, b)
print({k: round(v, 4) for k, v in row.items()})
print("weights cached; perceptual metrics load fine")
