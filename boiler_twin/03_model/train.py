"""
train.py — Full training pipeline for Boiler Digital Twin.

Phase A: Train one ControllerLSTM per loop (PC, LC, FC, TC, CC)
Phase B: Train PlantLSTM MIMO with scheduled sampling + causal loss

Usage:
  python train.py
  python train.py --phase controllers
  python train.py --phase plant
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))        # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_causal"))  # parse_dcs
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_data"))    # data_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from loops      import LOOPS, build_controllers
from plant      import PlantLSTM
from causal_loss import CausalLoss, combined_loss
from data_pipeline import build_datasets  # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Controller
CTRL_HIDDEN     = 128
CTRL_LAYERS     = 2
CTRL_DROPOUT    = 0.1
CTRL_LR         = 1e-3
CTRL_EPOCHS     = 100
CTRL_BATCH      = 256
CTRL_PATIENCE   = 15

# Plant
PLANT_HIDDEN    = 256
PLANT_LAYERS    = 2
PLANT_DROPOUT   = 0.1
PLANT_LR        = 1e-3
PLANT_EPOCHS    = 150
PLANT_BATCH     = 256
PLANT_PATIENCE  = 20
LAMBDA_CAUSAL   = 0.1
SS_START_EPOCH  = 10
SS_END_EPOCH    = 100
SS_MAX_RATIO    = 0.5

N_SCENARIOS     = 5
EMB_DIM         = 16

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "boiler_twin"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(model, path: Path, **meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **meta}, path)
    print(f"  Saved → {path}")


def _get_ss_ratio(epoch: int) -> float:
    if epoch < SS_START_EPOCH:
        return 0.0
    if epoch >= SS_END_EPOCH:
        return SS_MAX_RATIO
    progress = (epoch - SS_START_EPOCH) / (SS_END_EPOCH - SS_START_EPOCH)
    return progress * SS_MAX_RATIO


# ── Phase A: Train controllers ─────────────────────────────────────────────────

def train_controllers(data: dict):
    print("\n" + "=" * 60)
    print("PHASE A — Training Controller LSTMs")
    print("=" * 60)

    controllers = build_controllers(
        n_scenarios = N_SCENARIOS,
        hidden_dim  = CTRL_HIDDEN,
        num_layers  = CTRL_LAYERS,
        dropout     = CTRL_DROPOUT,
        emb_dim     = EMB_DIM,
    )

    trained = {}
    for ln, model in controllers.items():
        key_train = f"ctrl_{ln}_train"
        key_val   = f"ctrl_{ln}_val"
        if key_train not in data:
            print(f"  [{ln}] No data — skipping")
            continue

        model.to(DEVICE)
        tr_in, tr_tg, tr_sc = data[key_train]
        va_in, va_tg, va_sc = data[key_val]

        tr_ds = TensorDataset(
            torch.tensor(tr_in), torch.tensor(tr_tg), torch.tensor(tr_sc, dtype=torch.long))
        va_ds = TensorDataset(
            torch.tensor(va_in), torch.tensor(va_tg), torch.tensor(va_sc, dtype=torch.long))

        tr_dl = DataLoader(tr_ds, batch_size=CTRL_BATCH, shuffle=True,  pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=CTRL_BATCH * 2, pin_memory=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=CTRL_LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val, patience_cnt, best_state = float("inf"), 0, None
        ctrl_train_losses, ctrl_val_losses = [], []

        print(f"\n  [{ln}]  train={len(tr_ds):,}  val={len(va_ds):,}  "
              f"input_dim={tr_in.shape[-1]}")

        for epoch in range(CTRL_EPOCHS):
            # --- train ---
            model.train()
            train_loss = 0.0
            for x, y, sc in tr_dl:
                x, y, sc = x.to(DEVICE), y.to(DEVICE), sc.to(DEVICE)
                pred, _  = model(x, sc)
                loss     = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(x)
            train_loss /= len(tr_ds)

            # --- val ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y, sc in va_dl:
                    x, y, sc = x.to(DEVICE), y.to(DEVICE), sc.to(DEVICE)
                    pred, _  = model(x, sc)
                    val_loss += criterion(pred, y).item() * len(x)
            val_loss /= len(va_ds)
            scheduler.step(val_loss)
            ctrl_train_losses.append(train_loss)
            ctrl_val_losses.append(val_loss)

            if val_loss < best_val:
                best_val, patience_cnt = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1

            if epoch % 20 == 0 or patience_cnt >= CTRL_PATIENCE:
                print(f"    ep{epoch:3d}: train={train_loss:.5f}  val={val_loss:.5f}  "
                      f"pat={patience_cnt}")
            if patience_cnt >= CTRL_PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        _save(model, OUTPUT_DIR / "models" / f"ctrl_{ln}.pt",
              loop=ln, best_val=best_val)
        np.save(OUTPUT_DIR / f"ctrl_{ln}_train_losses.npy", np.array(ctrl_train_losses))
        np.save(OUTPUT_DIR / f"ctrl_{ln}_val_losses.npy",   np.array(ctrl_val_losses))
        trained[ln] = model
        print(f"  [{ln}] Best val loss: {best_val:.5f}")

    return trained


# ── Phase B: Train plant ───────────────────────────────────────────────────────

def train_plant(data: dict):
    print("\n" + "=" * 60)
    print("PHASE B — Training Plant LSTM (MIMO + Scheduled Sampling)")
    print("=" * 60)

    n_in  = data["n_plant_in"]
    n_out = data["n_pv"]

    model = PlantLSTM(
        n_in        = n_in,
        n_out       = n_out,
        hidden_dim  = PLANT_HIDDEN,
        num_layers  = PLANT_LAYERS,
        dropout     = PLANT_DROPOUT,
        n_scenarios = N_SCENARIOS,
        emb_dim     = EMB_DIM * 2,
    ).to(DEVICE)

    causal_fn = CausalLoss(lambda_causal=LAMBDA_CAUSAL).to(DEVICE)

    cv_tr, pi_tr, pte_tr, ptr_tr, sc_tr = data["plant_train"]
    cv_va, pi_va, pte_va, ptr_va, sc_va = data["plant_val"]

    tr_ds = TensorDataset(
        torch.tensor(cv_tr),  torch.tensor(pi_tr),
        torch.tensor(pte_tr), torch.tensor(ptr_tr),
        torch.tensor(sc_tr, dtype=torch.long))
    va_ds = TensorDataset(
        torch.tensor(cv_va),  torch.tensor(pi_va),
        torch.tensor(pte_va), torch.tensor(ptr_va),
        torch.tensor(sc_va, dtype=torch.long))

    tr_dl = DataLoader(tr_ds, batch_size=PLANT_BATCH, shuffle=True,  pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=PLANT_BATCH * 2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=PLANT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, PLANT_EPOCHS)

    best_val, patience_cnt, best_state = float("inf"), 0, None
    train_losses, val_losses = [], []

    print(f"  Plant input: {n_in}  PVs: {n_out}  "
          f"train={len(tr_ds):,}  val={len(va_ds):,}")

    for epoch in range(PLANT_EPOCHS):
        ss_ratio = _get_ss_ratio(epoch)

        # --- train ---
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for cv, pi, pte, ptr, sc in tr_dl:
            cv, pi, pte, ptr, sc = (
                cv.to(DEVICE), pi.to(DEVICE),
                pte.to(DEVICE), ptr.to(DEVICE), sc.to(DEVICE))

            pv_pred = model(cv, pi, sc, teacher_pv=pte, ss_ratio=ss_ratio)
            total, mse, causal = combined_loss(pv_pred, ptr, cv, causal_fn)

            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += total.item() * len(cv)
        train_loss /= len(tr_ds)
        train_losses.append(train_loss)

        # --- val (always autoregressive) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cv, pi, pte, ptr, sc in va_dl:
                cv, pi, ptr, sc = (
                    cv.to(DEVICE), pi.to(DEVICE),
                    ptr.to(DEVICE), sc.to(DEVICE))
                pv_pred  = model(cv, pi, sc, teacher_pv=None, ss_ratio=1.0)
                val_loss += nn.functional.mse_loss(pv_pred, ptr).item() * len(cv)
        val_loss /= len(va_ds)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val:
            best_val, patience_cnt = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or patience_cnt >= PLANT_PATIENCE:
            print(f"  ep{epoch:3d}: train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"ss={ss_ratio:.2f}  pat={patience_cnt}  "
                  f"({time.time()-t0:.1f}s)")
        if patience_cnt >= PLANT_PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    _save(model, OUTPUT_DIR / "models" / "plant.pt",
          n_in=n_in, n_out=n_out, best_val=best_val)
    print(f"\n✓ Plant model trained. Best val loss: {best_val:.5f}")

    np.save(OUTPUT_DIR / "plant_train_losses.npy", np.array(train_losses))
    np.save(OUTPUT_DIR / "plant_val_losses.npy",   np.array(val_losses))

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "controllers", "plant"],
                        default="all")
    parser.add_argument("--no-save-data", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    data = build_datasets(save=not args.no_save_data)

    if args.phase in ("all", "controllers"):
        trained_controllers = train_controllers(data)

    if args.phase in ("all", "plant"):
        trained_plant = train_plant(data)

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
