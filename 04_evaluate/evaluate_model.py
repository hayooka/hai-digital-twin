"""
evaluate_model.py — Regression evaluation for any GRUPlant checkpoint.

Prints tables for:
  - NRMSE per PV (overall, normal-only, attack-only)
  - NRMSE per PV per scenario
  - Mean NRMSE per scenario
  - Per-scenario ranking across PVs

Saves results to <ckpt_dir>/eval_results.json

Usage:
    python 04_evaluate/evaluate_model.py
    python 04_evaluate/evaluate_model.py --ckpt outputs/pipeline/gru_scenario_haiend
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant
from config import PV_COLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 128

SCENARIO_NAMES = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
PV_SHORT       = [p.replace("P1_", "") for p in PV_COLS]


# ── helpers ───────────────────────────────────────────────────────────────────

def nrmse(true: np.ndarray, pred: np.ndarray) -> float:
    if true.size == 0:
        return float("nan")
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    r    = float(true.max() - true.min())
    return 0.0 if r < 1e-10 else rmse / r


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred))) if true.size > 0 else float("nan")


def load_model(ckpt_dir: Path) -> GRUPlant:
    ckpt = torch.load(ckpt_dir / "gru_plant.pt", map_location=DEVICE)
    ms   = ckpt["model_state"]
    emb  = ms["scenario_emb.weight"].shape[1]
    model = GRUPlant(
        n_plant_in  = ckpt.get("n_plant_in",  ms["encoder.weight_ih_l0"].shape[1] - emb),
        n_pv        = ckpt.get("n_pv",        ms["fc.3.weight"].shape[0]),
        hidden      = ckpt["hidden"],
        layers      = ckpt["layers"],
        n_scenarios = ms["scenario_emb.weight"].shape[0],
        n_haiend    = ckpt.get("n_haiend", 0),
    ).to(DEVICE)
    model.load_state_dict(ms)
    model.eval()
    return model, ckpt.get("val_loss", float("nan"))


def run_inference(model, plant_data, split="test") -> np.ndarray:
    X, Xcv   = plant_data[f"X_{split}"], plant_data[f"X_cv_target_{split}"]
    pv_init  = plant_data[f"pv_init_{split}"]
    sc       = plant_data[f"scenario_{split}"]
    N, TL    = X.shape[0], plant_data[f"pv_target_{split}"].shape[1]
    preds    = np.zeros((N, TL, plant_data["n_pv"]), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl = slice(i, i + BATCH)
            pv_out, _ = model.predict(
                torch.tensor(X[sl]).float().to(DEVICE),
                torch.tensor(Xcv[sl]).float().to(DEVICE),
                torch.tensor(pv_init[sl]).float().to(DEVICE),
                torch.tensor(sc[sl]).long().to(DEVICE),
            )
            preds[i:i + pv_out.size(0)] = pv_out.cpu().numpy()
    return preds


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def hline(width=70):
    print("-" * width)


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(ckpt_dir: Path, data_dir: str = None):
    print(f"\nCheckpoint : {ckpt_dir}")
    model, val_loss = load_model(ckpt_dir)
    print(f"Val loss   : {val_loss:.6f}" if not np.isnan(val_loss) else "Val loss   : n/a")

    print("\nLoading data...")
    data       = load_and_prepare_data(data_dir)
    plant_data = data["plant"]
    pv_true    = plant_data["pv_target_test"]       # (N, T, 5)
    sc_arr     = plant_data["scenario_test"]         # (N,)

    print("Running inference...")
    preds = run_inference(model, plant_data)         # (N, T, 5)

    # ── 1. Overall NRMSE & MAE per PV ─────────────────────────────────────────
    section("1. Overall NRMSE & MAE per PV  (full test set)")
    col_w = 12
    header = f"{'PV':<12}" + "".join(f"{'NRMSE':>{col_w}}{'MAE':>{col_w}}" for _ in [""])
    print(f"{'PV':<12}{'NRMSE':>{col_w}}{'MAE':>{col_w}}")
    hline()
    overall_nrmse, overall_mae = {}, {}
    for k, pv in enumerate(PV_COLS):
        t = pv_true[:, :, k].flatten()
        p = preds[:, :, k].flatten()
        overall_nrmse[pv] = nrmse(t, p)
        overall_mae[pv]   = mae(t, p)
        print(f"{PV_SHORT[k]:<12}{overall_nrmse[pv]:>{col_w}.4f}{overall_mae[pv]:>{col_w}.4f}")
    hline()
    mean_n = np.mean(list(overall_nrmse.values()))
    mean_m = np.mean(list(overall_mae.values()))
    print(f"{'MEAN':<12}{mean_n:>{col_w}.4f}{mean_m:>{col_w}.4f}")

    # ── 2. NRMSE per PV — Normal vs Attack split ───────────────────────────────
    section("2. NRMSE per PV — Normal vs Attack windows")
    norm_mask   = sc_arr == 0
    attack_mask = sc_arr > 0
    print(f"  Normal windows : {norm_mask.sum()}   Attack windows : {attack_mask.sum()}")
    print()
    print(f"{'PV':<12}{'Normal':>{col_w}}{'Attack':>{col_w}}{'Δ (Atk-Nrm)':>{col_w+2}}")
    hline()
    normal_nrmse, attack_nrmse = {}, {}
    for k, pv in enumerate(PV_COLS):
        nt = nrmse(pv_true[norm_mask,   :, k].flatten(), preds[norm_mask,   :, k].flatten())
        at = nrmse(pv_true[attack_mask, :, k].flatten(), preds[attack_mask, :, k].flatten())
        normal_nrmse[pv] = nt
        attack_nrmse[pv] = at
        delta = at - nt
        print(f"{PV_SHORT[k]:<12}{nt:>{col_w}.4f}{at:>{col_w}.4f}{delta:>{col_w+2}.4f}")
    hline()
    print(f"{'MEAN':<12}"
          f"{np.mean(list(normal_nrmse.values())):>{col_w}.4f}"
          f"{np.mean(list(attack_nrmse.values())):>{col_w}.4f}")

    # ── 3. NRMSE per PV per scenario ──────────────────────────────────────────
    section("3. NRMSE per PV per Scenario")
    sc_nrmse = {}
    sc_ids   = [i for i in SCENARIO_NAMES if (sc_arr == i).sum() > 0]
    sc_labels = [SCENARIO_NAMES[i] for i in sc_ids]

    # header
    print(f"{'PV':<12}" + "".join(f"{s:>{col_w}}" for s in sc_labels))
    hline()
    pv_sc_nrmse = {}
    for k, pv in enumerate(PV_COLS):
        row = {}
        line = f"{PV_SHORT[k]:<12}"
        for sc_id in sc_ids:
            mask = sc_arr == sc_id
            v = nrmse(pv_true[mask, :, k].flatten(), preds[mask, :, k].flatten())
            row[SCENARIO_NAMES[sc_id]] = v
            line += f"{v:>{col_w}.4f}"
        pv_sc_nrmse[pv] = row
        print(line)
    hline()
    # mean per scenario
    line = f"{'MEAN':<12}"
    for sc_id in sc_ids:
        mask = sc_arr == sc_id
        vals = [nrmse(pv_true[mask, :, k].flatten(), preds[mask, :, k].flatten())
                for k in range(len(PV_COLS))]
        sc_nrmse[SCENARIO_NAMES[sc_id]] = np.mean(vals)
        line += f"{sc_nrmse[SCENARIO_NAMES[sc_id]]:>{col_w}.4f}"
    print(line)

    # ── 4. Attack scenario breakdown ──────────────────────────────────────────
    section("4. Attack Scenario Breakdown — Mean NRMSE vs Normal baseline")
    normal_mean = sc_nrmse.get("Normal", float("nan"))
    print(f"{'Scenario':<14}{'Mean NRMSE':>12}{'vs Normal':>12}{'Ratio':>10}")
    hline(50)
    for sc_id in sc_ids:
        if sc_id == 0:
            continue
        sc_name = SCENARIO_NAMES[sc_id]
        v       = sc_nrmse[sc_name]
        delta   = v - normal_mean
        ratio   = v / normal_mean if normal_mean > 0 else float("nan")
        print(f"{sc_name:<14}{v:>12.4f}{delta:>12.4f}{ratio:>10.2f}x")

    # ── summary ───────────────────────────────────────────────────────────────
    section("Summary")
    print(f"  Overall mean NRMSE : {mean_n:.4f}")
    print(f"  Normal mean NRMSE  : {np.mean(list(normal_nrmse.values())):.4f}")
    print(f"  Attack mean NRMSE  : {np.mean(list(attack_nrmse.values())):.4f}")

    # ── save JSON ─────────────────────────────────────────────────────────────
    out = {
        "checkpoint":           str(ckpt_dir),
        "val_loss":             float(val_loss) if not np.isnan(val_loss) else None,
        "overall_nrmse_per_pv": {pv: float(v) for pv, v in overall_nrmse.items()},
        "overall_mae_per_pv":   {pv: float(v) for pv, v in overall_mae.items()},
        "mean_nrmse":           float(mean_n),
        "normal_nrmse_per_pv":  {pv: float(v) for pv, v in normal_nrmse.items()},
        "attack_nrmse_per_pv":  {pv: float(v) for pv, v in attack_nrmse.items()},
        "nrmse_per_scenario":   {sc: float(v) for sc, v in sc_nrmse.items()},
        "nrmse_per_pv_per_scenario": {
            pv: {sc: float(v) for sc, v in row.items()}
            for pv, row in pv_sc_nrmse.items()
        },
    }
    out_path = ckpt_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/pipeline/gru_scenario_haiend")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to scaled_split data dir (default: outputs/scaled_split/)")
    args = parser.parse_args()
    evaluate(ROOT / args.ckpt, args.data_dir)
