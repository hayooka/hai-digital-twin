"""
rollout.py — Closed-loop generation for Boiler Digital Twin.

Given a starting point in the data and a scenario_id, runs the full
controller + plant loop autoregressively to generate synthetic sensor data
for any condition (normal or attack).

Usage:
  python rollout.py --scenario 0 --horizon 1800   # 30 min normal
  python rollout.py --scenario 1 --horizon 900    # 15 min AP attack
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_causal"))

from loops       import LOOPS, ControllerLSTM, build_controllers
from plant       import PlantLSTM
from parse_dcs   import LOOPS_DEF, PV_COLS, CV_COLS, PLANT_AUX_COLS
from data_pipeline import PLANT_IN_COLS, PLANT_OUT_COLS  # type: ignore

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "boiler_twin"
MODEL_DIR  = OUTPUT_DIR / "models"

WARMUP_STEPS = 300   # Run on real data to warm up LSTM hidden states


# ── Load trained models ───────────────────────────────────────────────────────

def load_controllers(device=DEVICE) -> dict[str, ControllerLSTM]:
    controllers = build_controllers()
    for ln, model in controllers.items():
        ckpt_path = MODEL_DIR / f"ctrl_{ln}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.to(device).eval()
        else:
            print(f"  [WARN] No checkpoint for {ln} — using random weights")
            model.to(device).eval()
    return controllers


def load_plant(device=DEVICE) -> PlantLSTM:
    ckpt  = torch.load(MODEL_DIR / "plant.pt", map_location=device)
    model = PlantLSTM(n_in=ckpt["n_in"], n_out=ckpt["n_out"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


# ── Warmup: initialise hidden states on real data ─────────────────────────────

def warmup_hidden_states(
    df_warmup,
    controllers: dict,
    plant:       PlantLSTM,
    scaler,
    sensor_cols: list[str],
    scenario_id: int,
    device=DEVICE,
):
    """Run warmup_steps of real data through all models to get hidden states."""
    sc_tensor = torch.tensor([scenario_id], dtype=torch.long, device=device)

    ctrl_hiddens = {}
    for ln, model in controllers.items():
        defn = LOOPS_DEF[ln]
        cols = [defn["sp"], defn["pv"]] + ([defn["cv_fb"]] if defn["cv_fb"] else []) + [defn["cv"]]
        cols = [c for c in cols if c in sensor_cols]

        col_idx  = [sensor_cols.index(c) for c in cols]
        raw      = df_warmup[sensor_cols].values.astype(np.float32)
        scaled   = scaler.transform(raw)[:, col_idx]

        x_warm   = torch.tensor(scaled[:-1, :-1], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, h = model(x_warm, sc_tensor)
        ctrl_hiddens[ln] = h

    # Plant warmup
    plant_cols = [c for c in PLANT_IN_COLS + PLANT_OUT_COLS if c in sensor_cols]
    p_idx      = [sensor_cols.index(c) for c in plant_cols]
    raw        = df_warmup[sensor_cols].values.astype(np.float32)
    scaled     = scaler.transform(raw)[:, p_idx]

    n_in   = len([c for c in PLANT_IN_COLS if c in sensor_cols])
    n_out  = len([c for c in PLANT_OUT_COLS if c in sensor_cols])
    plant_h = None

    sc_plant = torch.tensor([scenario_id], dtype=torch.long, device=device)
    with torch.no_grad():
        for t in range(len(df_warmup)):
            cv_t = torch.tensor(scaled[t, :n_in], dtype=torch.float32).unsqueeze(0).to(device)
            pv_t = torch.tensor(scaled[t, n_in:], dtype=torch.float32).unsqueeze(0).to(device)
            _, plant_h = plant.step(cv_t, pv_t, sc_plant, plant_h)

    pv_current = torch.tensor(
        scaled[-1, n_in:], dtype=torch.float32).unsqueeze(0).to(device)

    return ctrl_hiddens, plant_h, pv_current


# ── Closed-loop rollout ───────────────────────────────────────────────────────

def rollout(
    df,
    start_idx:   int,
    horizon:     int,
    scenario_id: int,
    controllers: dict,
    plant:       PlantLSTM,
    scaler,
    sensor_cols: list[str],
    device=DEVICE,
) -> dict:
    """
    Generate `horizon` steps of synthetic sensor data.

    Returns:
      pred_pvs  : (horizon, n_pv)   predicted PVs
      actual_pvs: (horizon, n_pv)   ground-truth PVs (for comparison)
      pred_cvs  : (horizon, n_cv)   predicted CVs from controllers
    """
    if start_idx + WARMUP_STEPS + horizon >= len(df):
        raise ValueError("Not enough data for warmup + horizon")

    df_warmup = df.iloc[start_idx:start_idx + WARMUP_STEPS]
    sc_tensor = torch.tensor([scenario_id], dtype=torch.long, device=device)

    ctrl_h, plant_h, pv_current = warmup_hidden_states(
        df_warmup, controllers, plant, scaler, sensor_cols, scenario_id, device)

    n_in   = len([c for c in PLANT_IN_COLS  if c in sensor_cols])
    n_out  = len([c for c in PLANT_OUT_COLS if c in sensor_cols])
    pv_col_idx = [sensor_cols.index(c) for c in PLANT_OUT_COLS if c in sensor_cols]
    cv_col_idx = {ln: sensor_cols.index(LOOPS_DEF[ln]["cv"])
                  for ln in LOOPS_DEF if LOOPS_DEF[ln]["cv"] in sensor_cols}

    pred_pvs, actual_pvs, pred_cvs = [], [], []

    for t in range(horizon):
        abs_t = start_idx + WARMUP_STEPS + t

        # ── Step 1: Run each controller → CV ─────────────────────────────────
        cvs_scaled = {}
        for ln, model in controllers.items():
            defn  = LOOPS_DEF[ln]
            sp_c  = defn["sp"]
            fb_c  = defn["cv_fb"]

            if sp_c not in sensor_cols:
                continue

            sp_idx = sensor_cols.index(sp_c)
            row_s  = scaler.transform(df.iloc[[abs_t]][sensor_cols].values.astype(np.float32))
            sp_val = torch.tensor([[row_s[0, sp_idx]]], dtype=torch.float32, device=device)

            # PV comes from plant prediction
            pv_loop_idx = PV_COLS.index(defn["pv"]) if defn["pv"] in PV_COLS else None
            pv_val = (pv_current[:, pv_loop_idx:pv_loop_idx + 1]
                      if pv_loop_idx is not None else torch.zeros(1, 1, device=device))

            if fb_c and fb_c in sensor_cols:
                fb_idx = sensor_cols.index(fb_c)
                fb_val = torch.tensor([[row_s[0, fb_idx]]], dtype=torch.float32, device=device)
                x_ctrl = torch.cat([sp_val, pv_val, fb_val], dim=-1)
            else:
                x_ctrl = torch.cat([sp_val, pv_val], dim=-1)

            with torch.no_grad():
                cv_pred, ctrl_h[ln] = model.step(x_ctrl, sc_tensor, ctrl_h[ln])
            cvs_scaled[ln] = cv_pred.item()

        # ── Step 2: Build plant input row ────────────────────────────────────
        row_raw    = df.iloc[abs_t][sensor_cols].values.astype(np.float32)
        row_scaled = scaler.transform(row_raw.reshape(1, -1))[0]

        # Override CV values with controller predictions
        for ln, cv_val in cvs_scaled.items():
            if ln in cv_col_idx:
                row_scaled[cv_col_idx[ln]] = cv_val

        plant_in_idx = [sensor_cols.index(c) for c in PLANT_IN_COLS if c in sensor_cols]
        cv_aux_t = torch.tensor(
            row_scaled[plant_in_idx], dtype=torch.float32).unsqueeze(0).to(device)

        # ── Step 3: Plant step → PVs ──────────────────────────────────────────
        with torch.no_grad():
            pv_pred, plant_h = plant.step(cv_aux_t, pv_current, sc_tensor, plant_h)

        pv_current = pv_pred.detach()

        # Collect results (inverse transform for interpretability)
        full_row = np.zeros((1, len(sensor_cols)), dtype=np.float32)
        for i, ci in enumerate(pv_col_idx):
            full_row[0, ci] = pv_pred[0, i].item()
        pv_raw = scaler.inverse_transform(full_row)[0, pv_col_idx]

        actual_row = df.iloc[abs_t + 1][sensor_cols].values.astype(np.float32)
        actual_raw = scaler.inverse_transform(actual_row.reshape(1, -1))[0, pv_col_idx]

        pred_pvs.append(pv_raw)
        actual_pvs.append(actual_raw)
        pred_cvs.append(np.array([cvs_scaled.get(ln, 0.0) for ln in LOOPS_DEF]))

    return {
        "pred_pvs":   np.array(pred_pvs),    # (horizon, n_pv)
        "actual_pvs": np.array(actual_pvs),  # (horizon, n_pv)
        "pred_cvs":   np.array(pred_cvs),    # (horizon, n_loops)
        "pv_cols":    [c for c in PLANT_OUT_COLS if c in sensor_cols],
        "scenario":   scenario_id,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=0,
                        help="0=normal, 1=AP_no, 2=AP_with, 3=AE_no, 4=AE_with")
    parser.add_argument("--horizon",  type=int, default=1800,
                        help="Steps to generate (seconds at 1Hz)")
    parser.add_argument("--start",    type=int, default=0,
                        help="Start index in test data")
    args = parser.parse_args()

    import joblib
    import pandas as pd
    from pathlib import Path

    base = Path(__file__).resolve().parents[2]
    scaler      = joblib.load(OUTPUT_DIR / "scaler.pkl")
    sensor_cols = list(np.load(OUTPUT_DIR / "scaler_cols.npy", allow_pickle=True))
    df_test     = pd.read_csv(base / "data" / "processed" / "train4.csv", low_memory=False)

    controllers = load_controllers()
    plant       = load_plant()

    print(f"\nGenerating {args.horizon}s of data | scenario={args.scenario}")
    results = rollout(
        df         = df_test,
        start_idx  = args.start,
        horizon    = args.horizon,
        scenario_id= args.scenario,
        controllers= controllers,
        plant      = plant,
        scaler     = scaler,
        sensor_cols= sensor_cols,
    )

    out_path = OUTPUT_DIR / f"generated_sc{args.scenario}_h{args.horizon}.npz"
    np.savez_compressed(out_path, **results)
    print(f"✓ Saved generated data → {out_path}")

    # Quick summary
    pred, actual = results["pred_pvs"], results["actual_pvs"]
    rmse = np.sqrt(((pred - actual) ** 2).mean(axis=0))
    print("\nRMSE per PV:")
    for col, r in zip(results["pv_cols"], rmse):
        print(f"  {col}: {r:.4f}")


if __name__ == "__main__":
    main()
