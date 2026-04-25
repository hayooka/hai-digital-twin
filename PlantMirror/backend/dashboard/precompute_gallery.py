"""
precompute_gallery.py - pre-render a grid of attacks for the dashboard gallery.

Runs once, saves NPZ + JSON into ./cache/, then the Streamlit app reads them
instantly on load. Safe to kill mid-run; partial results are still usable.

Usage (from the dashboard/ directory):
    python precompute_gallery.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "generator"))
sys.path.insert(0, str(REPO / "attack_sim"))

from core import load_bundle, load_replay, default_paths  # noqa: E402
from attacks import AttackSpec, InjectionPoint, AttackType, run_attack_sim  # noqa: E402


# Grid: 5 loops x 3 injection points x 2 attack types = 30 attacks
LOOPS = ["PC", "LC", "FC", "TC", "CC"]
INJECTIONS = [InjectionPoint.SP, InjectionPoint.CV, InjectionPoint.PV]
TYPES = [AttackType.BIAS, AttackType.FREEZE]

# Per-loop magnitude (physical units, rough midpoints of real ops range)
MAG_BY_LOOP = {"PC": 0.05, "LC": 20.0, "FC": 50.0, "TC": 2.0, "CC": 2.0}

CURSOR = 1000
START_OFFSET = -30
DURATION = 90


def main() -> None:
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)

    t0 = time.time()
    bundle = load_bundle(*map(lambda k: default_paths()[k], ["ckpt_dir", "split_dir"]))
    paths = default_paths()
    src = load_replay(paths["test_csvs"][0], bundle.scalers)
    print(f"Bundle + replay loaded in {time.time()-t0:.1f}s")

    baselines, attacked, labels, index = [], [], [], []
    t_run = time.time()
    for loop in LOOPS:
        mag = MAG_BY_LOOP[loop]
        for ip in INJECTIONS:
            for at in TYPES:
                spec = AttackSpec(
                    target_loop=loop,
                    injection_point=ip,
                    attack_type=at,
                    start_offset=START_OFFSET,
                    duration=DURATION,
                    magnitude=mag if at == AttackType.BIAS else 0.0,
                )
                r = run_attack_sim(bundle, src, t_end=CURSOR, spec=spec, scenario=0)
                if r is None:
                    print(f"  SKIP  {loop}/{ip.value}/{at.value}  (cursor out of range)")
                    continue
                baselines.append(r.baseline["pv_physical"].astype(np.float32))
                attacked.append(r.attacked["pv_physical"].astype(np.float32))
                labels.append(r.attack_label.astype(np.int8))
                index.append({
                    "loop": loop,
                    "injection_point": ip.value,
                    "attack_type": at.value,
                    "magnitude": spec.magnitude,
                    "start_offset": spec.start_offset,
                    "duration": spec.duration,
                    "peak_dpv": float(np.max(np.abs(
                        r.attacked["pv_physical"] - r.baseline["pv_physical"]))),
                })
                print(f"  OK    {loop}/{ip.value:2s}/{at.value:6s}  "
                      f"peakDPV={index[-1]['peak_dpv']:>7.3f}")

    if not baselines:
        print("No attacks succeeded.")
        return

    B = np.stack(baselines, axis=0)
    A = np.stack(attacked, axis=0)
    L = np.stack(labels, axis=0)
    np.savez_compressed(
        cache / "attack_gallery.npz",
        baseline_pv=B, attacked_pv=A, attack_label=L,
    )
    with open(cache / "attack_gallery_index.json", "w") as f:
        json.dump(
            {"items": index,
             "meta": {"cursor": CURSOR, "start_offset": START_OFFSET,
                      "duration": DURATION, "n": len(index)}},
            f, indent=2,
        )
    print(f"\nRendered {len(index)} attacks in {time.time()-t_run:.1f}s total")
    print(f"Saved to {cache}")


if __name__ == "__main__":
    main()
