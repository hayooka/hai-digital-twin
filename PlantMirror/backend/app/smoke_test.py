"""
smoke_test.py — headless end-to-end verification of the HAI Digital Twin app.

Exercises every piece that the Streamlit UI calls into:
- Bundle loading (plant + 5 controllers + scalers)
- Replay loading for both test CSVs
- Plant prediction on a normal window and an attack window
- Closed-loop rollout under every scenario label
- Closed-loop rollout with SP overrides
- Causal-graph load + rank_suspects for each PV

Exits 1 on the first failure. Designed to be run before every push.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

from causal_utils import CausalGraph
from twin_core import (
    INPUT_LEN,
    PV_COLS,
    TARGET_LEN,
    build_plant_window,
    closed_loop_rollout,
    default_paths,
    load_bundle,
    load_replay,
    per_pv_mse,
    per_step_residual,
    predict_plant,
    window_mse,
)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    paths = default_paths()
    print("== HAI Digital Twin smoke test ==")

    # 1. Load bundle ----------------------------------------------------------
    try:
        bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
        assert bundle.plant.n_plant_in == 128
        assert bundle.plant.n_pv == 5
        assert set(bundle.controllers) == {"PC", "LC", "FC", "TC", "CC"}
        _ok(f"bundle loaded on {bundle.device}")
    except Exception as e:
        traceback.print_exc()
        _fail(f"bundle load: {e}")

    # 2. Causal graph ---------------------------------------------------------
    try:
        graph = CausalGraph.load(paths["causal_json"])
        pvs_in_graph = sum(1 for pv in PV_COLS if pv in graph.parents)
        assert pvs_in_graph >= 4, f"only {pvs_in_graph} PVs in graph"
        _ok(f"causal graph loaded ({pvs_in_graph}/{len(PV_COLS)} PVs present)")
    except Exception as e:
        traceback.print_exc()
        _fail(f"causal graph: {e}")

    # 3. Both replay CSVs -----------------------------------------------------
    for csv_path in paths["test_csvs"]:
        if not csv_path.exists():
            print(f"  [skip] {csv_path} missing")
            continue
        try:
            src = load_replay(csv_path, bundle.scalers)
            assert src.scaled.shape[1] == 133
            assert len(src) > INPUT_LEN + TARGET_LEN
            _ok(f"replay '{src.name}' ({len(src):,} rows)")
        except Exception as e:
            traceback.print_exc()
            _fail(f"replay load {csv_path}: {e}")

        # 4. Plant predict on two windows (normal + any attack-containing) ----
        windows_to_test = [INPUT_LEN]
        if "label" in src.df_raw.columns:
            attack_idx = np.where(src.df_raw["label"].to_numpy() > 0)[0]
            if len(attack_idx):
                mid = int(attack_idx[len(attack_idx) // 2])
                t = max(INPUT_LEN, min(mid, len(src) - TARGET_LEN - 1))
                if t != INPUT_LEN:
                    windows_to_test.append(t)

        for t_end in windows_to_test:
            win = build_plant_window(bundle, src, t_end)
            if win is None:
                _fail(f"build_plant_window returned None at t_end={t_end}")
            pv_pred = predict_plant(bundle, win)
            pv_true = win["pv_target"].squeeze(0).cpu().numpy()
            mse = window_mse(pv_pred, pv_true)
            per_pv = per_pv_mse(pv_pred, pv_true)
            per_step = per_step_residual(pv_pred, pv_true)
            assert pv_pred.shape == (TARGET_LEN, 5)
            assert per_pv.shape == (5,)
            assert per_step.shape == (TARGET_LEN,)
            _ok(f"predict @ t_end={t_end:6d}  mse={mse:.5f}  "
                f"{'ANOMALY' if mse > bundle.threshold else 'normal'}")

        # 5. Closed-loop rollout under every scenario ------------------------
        for sc in range(4):
            out = closed_loop_rollout(bundle, src, INPUT_LEN, scenario=sc)
            if out is None:
                _fail(f"closed_loop scenario={sc} returned None")
            assert out["pv_physical"].shape == (TARGET_LEN, 5)
            assert not np.isnan(out["pv_physical"]).any(), f"NaN in scenario {sc}"
        _ok("closed-loop rollout ran for all 4 scenarios")

        # 6. Closed-loop with SP overrides -----------------------------------
        # Override PC setpoint by +10% to make sure the branch is exercised
        sp_col = bundle.scalers.ctrl["PC"]["cols"][0]  # type: ignore[index]
        cur = float(src.df_raw[sp_col].iloc[INPUT_LEN - 1])
        override = {"PC": cur * 1.1 if cur != 0 else 1.0}
        out = closed_loop_rollout(
            bundle, src, INPUT_LEN, sp_overrides=override, scenario=0
        )
        if out is None:
            _fail("closed_loop with sp_overrides returned None")
        _ok("closed-loop rollout with SP override")

    # 7. Causal rank_suspects per PV -----------------------------------------
    for pv in PV_COLS:
        if pv not in graph.parents:
            continue
        suspects = graph.rank_suspects(pv, max_depth=3)
        assert len(suspects) > 0, f"no suspects for {pv}"
    _ok("rank_suspects returns non-empty for PVs present in graph")

    print("== all checks passed ==")


if __name__ == "__main__":
    main()
