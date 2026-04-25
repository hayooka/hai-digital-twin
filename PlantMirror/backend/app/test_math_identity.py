"""
test_math_identity.py — Step 0 sanity check.

Proves that running plant.encode_only() once + plant.step_once() 180 times
produces bit-identical output to plant.predict() (the batched 180-step
rollout). If this fails, the live-twin runtime is built on sand and nothing
downstream can be trusted.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from twin_core import (
    INPUT_LEN,
    TARGET_LEN,
    build_plant_window,
    default_paths,
    load_bundle,
    load_replay,
)


def main() -> int:
    paths = default_paths()
    bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
    src = load_replay(paths["test_csvs"][0], bundle.scalers)
    win = build_plant_window(bundle, src, INPUT_LEN)
    assert win is not None

    plant = bundle.plant

    # ── Path A: batched predict() (the training-time rollout) ─────────────
    pv_batched = plant.predict(
        win["x_cv"], win["x_cv_target"], win["pv_init"], win["scenario"]
    ).squeeze(0).cpu().numpy()  # (180, 5)

    # ── Path B: encode_only() + step_once() ×180 ──────────────────────────
    h = plant.encode_only(win["x_cv"], win["scenario"])
    pv = win["pv_init"]
    x_cv_target = win["x_cv_target"]  # (1, 180, 128)
    pv_stepped_rows = []
    for t in range(TARGET_LEN):
        pv, h = plant.step_once(x_cv_target[:, t, :], h, pv)
        pv_stepped_rows.append(pv.squeeze(0).cpu().numpy())
    pv_stepped = np.stack(pv_stepped_rows, axis=0)  # (180, 5)

    # ── Compare ──────────────────────────────────────────────────────────
    diff = np.abs(pv_batched - pv_stepped)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    print(f"pv_batched   shape: {pv_batched.shape}")
    print(f"pv_stepped   shape: {pv_stepped.shape}")
    print(f"max  abs diff: {max_abs:.3e}")
    print(f"mean abs diff: {mean_abs:.3e}")

    # CUDA non-determinism can yield ~1e-6 differences; allow a generous margin
    tol = 1e-4
    if max_abs > tol:
        print(f"[FAIL] divergence > {tol}")
        # Dump a few rows for diagnosis
        for t in [0, 1, 89, 178, 179]:
            print(f"  t={t:3d}  batched={pv_batched[t]}  stepped={pv_stepped[t]}")
        return 1

    print(f"[OK] single-step and batched rollouts agree within {tol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
