"""
run_long_horizon_eval.py - evaluate the 1800-s plant (v2_weighted_init_best.pt)
on test1+test2 windows and produce per-PV / per-scenario NRMSE.

Open-loop evaluation: plant takes the REAL plant-input trajectory for the next
1800 s (not controller-predicted), so we measure plant forecast accuracy in
isolation from controller drift. This matches how the v2 plant was trained
(target_len=1800).

Saves -> cache/eval_1800s.json with the same key shape as eval_results.json
plus the new horizon. Tab 2 (Rollout Tester) reads this when present.

Run:
    python dashboard/run_long_horizon_eval.py
    python dashboard/run_long_horizon_eval.py --max-windows-per-csv 50  # quick
    python dashboard/run_long_horizon_eval.py --all                     # full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "generator"))

from core import (  # noqa: E402
    GRUPlant, default_paths, load_scalers, INPUT_LEN, PV_COLS,
)


V2_CKPT   = REPO / "training" / "checkpoints" / "v2_weighted_init_best.pt"
TEST1_CSV = Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test1.csv")
TEST2_CSV = Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test2.csv")
OUT_JSON  = HERE / "cache" / "eval_1800s.json"

TARGET_LEN_LONG = 1800
STRIDE = 600   # 10-min stride between evaluation windows (keeps runtime reasonable)

SCEN_NAMES = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}


def _load_v2_plant(device: torch.device) -> GRUPlant:
    blob = torch.load(V2_CKPT, map_location="cpu", weights_only=False)
    state = blob["model_state"]
    emb_dim = state["scenario_emb.weight"].shape[1]
    n_scen  = blob.get("n_scenarios", state["scenario_emb.weight"].shape[0])
    n_in    = blob.get("n_plant_in", state["encoder.weight_ih_l0"].shape[1] - emb_dim)
    n_pv    = blob.get("n_pv", state["fc.3.weight"].shape[0])
    plant = GRUPlant(
        n_plant_in=n_in, n_pv=n_pv,
        hidden=blob["hidden"], layers=blob["layers"],
        n_scenarios=n_scen, emb_dim=emb_dim,
        n_haiend=blob.get("n_haiend", 0),
    )
    plant.load_state_dict(state)
    plant.eval().to(device)
    return plant


def _scenario_for_window(label_arr: np.ndarray) -> int:
    """Window scenario = dominant non-zero label if any attack is present, else Normal.

    1800-s windows are mostly Normal with brief attack stretches inside; majority-vote
    would drown out every attack scenario. Labelling the window by *which kind of
    attack appears* gives meaningful per-scenario rows.
    """
    if label_arr.size == 0:
        return 0
    nonzero = label_arr[label_arr > 0]
    if nonzero.size == 0:
        return 0
    classes, counts = np.unique(nonzero, return_counts=True)
    return int(classes[counts.argmax()])


def _eval_csv(
    csv_path: Path, plant: GRUPlant, scalers, device: torch.device,
    max_windows: int | None,
) -> dict:
    """Run the plant on every window of (INPUT_LEN + TARGET_LEN_LONG) at STRIDE."""
    df_raw = pd.read_csv(csv_path, low_memory=False)
    scaled = scalers.scale_plant_row(df_raw)            # (T, 133)
    pv_idx = scalers.pv_idx
    plant_in_idx = scalers.plant_in_idx
    label = (df_raw["label"].astype(np.int8).to_numpy()
             if "label" in df_raw.columns else np.zeros(len(df_raw), dtype=np.int8))

    T = len(df_raw)
    starts = list(range(INPUT_LEN, T - TARGET_LEN_LONG, STRIDE))
    if max_windows is not None:
        starts = starts[:max_windows]
    print(f"  {csv_path.name}: {len(starts)} windows of {TARGET_LEN_LONG} s")

    # Per-PV squared-error and abs-target accumulators, segmented by scenario.
    per_pv_sse: dict[str, np.ndarray] = {pv: np.zeros(4) for pv in PV_COLS}
    per_pv_n:   dict[str, np.ndarray] = {pv: np.zeros(4) for pv in PV_COLS}
    pv_range:   dict[str, np.ndarray] = {pv: np.zeros(4) for pv in PV_COLS}  # max-min per scen

    t0 = time.time()
    for w_i, t_end in enumerate(starts):
        # Inputs
        x_cv     = scaled[t_end - INPUT_LEN:t_end, plant_in_idx]
        x_cv_tgt = scaled[t_end:t_end + TARGET_LEN_LONG, plant_in_idx]
        pv_init  = scaled[t_end - 1, pv_idx]
        pv_true  = scaled[t_end:t_end + TARGET_LEN_LONG, pv_idx]
        scen     = _scenario_for_window(label[t_end:t_end + TARGET_LEN_LONG])

        x_cv_t   = torch.from_numpy(x_cv).unsqueeze(0).float().to(device)
        x_cv_tgt_t = torch.from_numpy(x_cv_tgt).unsqueeze(0).float().to(device)
        pv_init_t  = torch.from_numpy(pv_init).unsqueeze(0).float().to(device)
        scen_t     = torch.tensor([scen], dtype=torch.long, device=device)

        pv_pred = plant.predict(x_cv_t, x_cv_tgt_t, pv_init_t, scen_t).squeeze(0).cpu().numpy()
        # Inverse-scale to physical units before NRMSE so it's comparable
        pv_pred_phys = scalers.inverse_plant(pv_pred, PV_COLS)
        pv_true_phys = scalers.inverse_plant(pv_true, PV_COLS)

        for i, pv in enumerate(PV_COLS):
            err  = pv_pred_phys[:, i] - pv_true_phys[:, i]
            sse  = float((err ** 2).sum())
            n    = int(err.size)
            per_pv_sse[pv][scen] += sse
            per_pv_n[pv][scen]   += n
            r = float(pv_true_phys[:, i].max() - pv_true_phys[:, i].min())
            if r > pv_range[pv][scen]:
                pv_range[pv][scen] = r

        if (w_i + 1) % 10 == 0 or w_i == len(starts) - 1:
            elapsed = time.time() - t0
            rate = (w_i + 1) / max(elapsed, 1e-3)
            print(f"    [{w_i + 1}/{len(starts)}]  {elapsed:.0f}s elapsed  ({rate:.1f}/s)")

    return {"per_pv_sse": per_pv_sse, "per_pv_n": per_pv_n, "pv_range": pv_range}


def _aggregate(results: list[dict]) -> dict:
    """Combine multiple CSV results -> per-PV per-scenario NRMSE."""
    per_pv_sse = {pv: np.zeros(4) for pv in PV_COLS}
    per_pv_n   = {pv: np.zeros(4) for pv in PV_COLS}
    pv_range   = {pv: np.zeros(4) for pv in PV_COLS}
    for r in results:
        for pv in PV_COLS:
            per_pv_sse[pv] += r["per_pv_sse"][pv]
            per_pv_n[pv]   += r["per_pv_n"][pv]
            pv_range[pv]   = np.maximum(pv_range[pv], r["pv_range"][pv])

    nrmse_per_pv_per_scen: dict = {}
    for pv in PV_COLS:
        nrmse_per_pv_per_scen[pv] = {}
        for s in range(4):
            n   = per_pv_n[pv][s]
            r   = pv_range[pv][s]
            if n > 0 and r > 0:
                rmse  = np.sqrt(per_pv_sse[pv][s] / n)
                nrmse = float(rmse / r)
            else:
                nrmse = None
            nrmse_per_pv_per_scen[pv][SCEN_NAMES[s]] = nrmse

    # Overall per-PV (collapse scenarios)
    overall_per_pv: dict = {}
    for pv in PV_COLS:
        n_tot = per_pv_n[pv].sum()
        if n_tot == 0:
            overall_per_pv[pv] = None
            continue
        rmse  = np.sqrt(per_pv_sse[pv].sum() / n_tot)
        # Use the max range across scenarios as the normaliser (consistent with per-scen)
        r_norm = max(pv_range[pv].max(), 1e-9)
        overall_per_pv[pv] = float(rmse / r_norm)

    # Mean per scenario (across PVs that have data)
    nrmse_per_scen: dict = {}
    for s in range(4):
        vals = [nrmse_per_pv_per_scen[pv][SCEN_NAMES[s]]
                for pv in PV_COLS
                if nrmse_per_pv_per_scen[pv][SCEN_NAMES[s]] is not None]
        nrmse_per_scen[SCEN_NAMES[s]] = float(np.mean(vals)) if vals else None

    return {
        "horizon_s": TARGET_LEN_LONG,
        "stride_s": STRIDE,
        "checkpoint": str(V2_CKPT),
        "overall_nrmse_per_pv": overall_per_pv,
        "nrmse_per_pv_per_scenario": nrmse_per_pv_per_scen,
        "nrmse_per_scenario": nrmse_per_scen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows-per-csv", type=int, default=50,
                    help="Cap windows per CSV. 50 keeps full run under ~5 min on GPU. "
                         "Use --all for the complete sweep.")
    ap.add_argument("--all", action="store_true", help="Process every window (slow).")
    args = ap.parse_args()

    if not V2_CKPT.exists():
        print(f"ERROR: v2 checkpoint not found at {V2_CKPT}", file=sys.stderr)
        sys.exit(1)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = default_paths()
    scalers = load_scalers(paths["split_dir"])
    print(f"Loading v2 plant from {V2_CKPT.name}...")
    plant = _load_v2_plant(device)
    n_params = sum(p.numel() for p in plant.parameters())
    print(f"  loaded: {n_params:,} params")

    max_w = None if args.all else args.max_windows_per_csv
    print(f"Max windows per CSV: {'ALL' if max_w is None else max_w}")
    print()

    results = []
    for csv in [TEST1_CSV, TEST2_CSV]:
        if not csv.exists():
            print(f"  skipping (not found): {csv}")
            continue
        print(f"Evaluating {csv.name}...")
        results.append(_eval_csv(csv, plant, scalers, device, max_w))

    if not results:
        print("ERROR: no CSVs processed.", file=sys.stderr)
        sys.exit(1)

    print("\nAggregating...")
    out = _aggregate(results)

    print("\nResults:")
    print(f"  overall_nrmse_per_pv: ", json.dumps(out["overall_nrmse_per_pv"], indent=2, default=float))
    print(f"  nrmse_per_scenario:   ", json.dumps(out["nrmse_per_scenario"],  indent=2, default=float))

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
