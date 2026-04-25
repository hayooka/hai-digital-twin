"""
train_plant_long.py — Retrain the GRU plant at a longer rollout horizon.

One-file, self-contained training script.  Reads the existing scaler/metadata
so the resulting checkpoint is drop-in compatible with
`hai-digital-twin/generator/core.py:load_bundle()`.

What it does
────────────
1. Loads the 133-column plant StandardScaler + sensor_cols from the existing
   generator scalers/metadata.
2. Reads the HAI processed CSVs, selects those 133 columns, scales in memory.
3. Builds sliding (input_len=300) + (target_len=USER) windows with stride 60,
   drops any window that contains NaNs or labelled-attack rows (treated as
   scenario=0 normal training data).
4. Trains a GRUPlant (2×512 hidden, 128 CV in → 5 PV out, scenario emb 4×32)
   with scheduled sampling.
5. Saves `gru_plant.pt` in the exact dict format core.py expects.

Horizon note
────────────
The current shipped checkpoint was trained at target_len=180 (3 min).  Pass
`--target_len 1800` to train a 30-minute version.  Longer targets need more
GPU memory (autoregressive unroll); drop --batch if you OOM.

vast.ai quickstart
──────────────────
  1. Pick an instance: pytorch/pytorch:2.4.0-cuda12.1 (or similar), 24 GB+ VRAM
     (A5000 / A100 for 30-min horizon, RTX 4090 fine for 10-min).
  2. Upload this repo:
         rsync -avz hai-digital-twin/ user@<host>:~/hai-digital-twin/
         rsync -avz processed/        user@<host>:~/processed/
  3. SSH in:
         cd ~/hai-digital-twin/training
         pip install -r requirements.txt
  4. Train (30-min horizon, fine-tune from existing weights):
         python train_plant_long.py \\
             --processed_dir ~/processed \\
             --scaler_dir    ../generator/scalers \\
             --init_from     ../generator/weights/gru_plant.pt \\
             --target_len    1800 \\
             --epochs        40 \\
             --batch         8 \\
             --output_dir    ./out_1800
  5. Pull the weights back:
         rsync -avz user@<host>:~/hai-digital-twin/training/out_1800/ ./out_1800/
         cp out_1800/gru_plant.pt ../generator/weights/
     Refresh the generator app — new horizon active.

Notes
─────
- Controllers are NOT retrained here.  If you bump target_len past 180, the
  generator's `closed_loop_rollout` will still call `controller.predict(target_len=TARGET_LEN)`;
  the controllers are GRU encoder-decoders and will autoregress at any length,
  but their quality degrades past their training horizon (180 s).  For an
  honest 30-min rollout, retrain controllers too (see train_ctrl_long.py).
- `--init_from` is optional; omit it to train from scratch.
- Only `scenario=0` (normal) windows are used during training here.  To re-add
  attack-conditioned training you'd need to merge labelled test CSVs.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── PV columns (must match LOOP_SPECS in generator/core.py) ─────────────────

PV_COLS: List[str] = ["P1_PIT01", "P1_LIT01", "P1_FT03Z", "P1_TIT01", "P1_TIT03"]


# ── Model definition — identical structure to generator/core.py GRUPlant ────

class GRUPlant(nn.Module):
    """MIMO GRU plant; encoder over [x_cv, scenario_emb] → decoder autoregresses PV."""

    def __init__(self, n_plant_in: int = 128, n_pv: int = 5,
                 hidden: int = 512, layers: int = 2,
                 n_scenarios: int = 4, emb_dim: int = 32,
                 n_haiend: int = 0, dropout: float = 0.1):
        super().__init__()
        self.n_plant_in = n_plant_in
        self.n_pv = n_pv
        self.n_haiend = n_haiend
        drop = dropout if layers > 1 else 0.0

        self.scenario_emb = nn.Embedding(n_scenarios, emb_dim)
        self.encoder = nn.GRU(n_plant_in + emb_dim, hidden, layers,
                              batch_first=True, dropout=drop)
        self.decoder = nn.GRU(n_plant_in + n_pv, hidden, layers,
                              batch_first=True, dropout=drop)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, n_pv),
        )
        if n_haiend > 0:
            self.haiend_head = nn.Sequential(
                nn.Linear(hidden, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, n_haiend),
            )
        else:
            self.haiend_head = None

    def forward(self, x_cv, x_cv_target, pv_init, scenario,
                pv_teacher=None, ss_ratio: float = 0.0):
        B, T, _ = x_cv_target.shape
        emb = self.scenario_emb(scenario).unsqueeze(1).expand(-1, x_cv.size(1), -1)
        _, h = self.encoder(torch.cat([x_cv, emb], dim=-1))

        pv = pv_init
        outs = []
        for t in range(T):
            dec_in = torch.cat([x_cv_target[:, t, :], pv], dim=-1).unsqueeze(1)
            step, h = self.decoder(dec_in, h)
            pv_pred = self.fc(step.squeeze(1))
            outs.append(pv_pred)

            if pv_teacher is not None and ss_ratio > 0.0:
                use_pred = (torch.rand(B, device=x_cv.device) < ss_ratio
                            ).float().unsqueeze(-1)
                pv = use_pred * pv_pred + (1 - use_pred) * pv_teacher[:, t, :]
            elif pv_teacher is not None:
                pv = pv_teacher[:, t, :]
            else:
                pv = pv_pred
        return torch.stack(outs, dim=1)


# ── Data prep ───────────────────────────────────────────────────────────────

@dataclass
class ScalerBundle:
    mean: np.ndarray
    scale: np.ndarray
    sensor_cols: List[str]
    plant_in_idx: np.ndarray
    pv_idx: np.ndarray


def load_scaler(scaler_dir: Path) -> ScalerBundle:
    s = joblib.load(scaler_dir / "scaler.pkl")
    with open(scaler_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    sensor_cols: List[str] = list(meta["sensor_cols"])
    pv_idx = np.array([sensor_cols.index(p) for p in PV_COLS], dtype=np.int64)
    plant_in_idx = np.array(
        [i for i, c in enumerate(sensor_cols) if c not in PV_COLS], dtype=np.int64,
    )
    return ScalerBundle(
        mean=s.mean_.astype(np.float32),
        scale=s.scale_.astype(np.float32),
        sensor_cols=sensor_cols, plant_in_idx=plant_in_idx, pv_idx=pv_idx,
    )


def read_csv_scaled(path: Path, scalers: ScalerBundle) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scaled_matrix (T, 133), label (T,)).  Skips NaN rows."""
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in scalers.sensor_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name}: missing columns {missing[:5]} (+{len(missing)-5} more)"
                           if len(missing) > 5 else f"{path.name}: missing {missing}")
    arr = df[scalers.sensor_cols].to_numpy(dtype=np.float32)
    lbl = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else np.zeros(len(arr), dtype=np.int64)
    scaled = (arr - scalers.mean) / scalers.scale
    # Drop rows with any NaN in scaled features (simple policy).
    finite = np.isfinite(scaled).all(axis=1)
    return scaled[finite], lbl[finite]


def build_windows(scaled: np.ndarray, labels: np.ndarray,
                  input_len: int, target_len: int, stride: int,
                  drop_attack: bool = True) -> np.ndarray:
    """
    Returns window start indices `t0` such that window covers [t0, t0+input_len+target_len).
    Each window is kept only if all its rows have label == 0 (normal) and the
    slice is fully inside `scaled`.
    """
    horizon = input_len + target_len
    starts = []
    for t0 in range(0, len(scaled) - horizon + 1, stride):
        if drop_attack and labels[t0:t0 + horizon].any():
            continue
        starts.append(t0)
    return np.array(starts, dtype=np.int64)


class PlantDataset(Dataset):
    def __init__(self, scaled: np.ndarray, starts: np.ndarray,
                 scalers: ScalerBundle, input_len: int, target_len: int,
                 scenario: int = 0):
        self.scaled = scaled
        self.starts = starts
        self.scalers = scalers
        self.input_len = input_len
        self.target_len = target_len
        self.scenario = scenario

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        t0 = int(self.starts[i])
        t1 = t0 + self.input_len
        t2 = t1 + self.target_len
        window = self.scaled[t0:t2]
        x_cv = window[:self.input_len][:, self.scalers.plant_in_idx]
        x_cv_target = window[self.input_len:][:, self.scalers.plant_in_idx]
        pv_init = window[self.input_len - 1][self.scalers.pv_idx]
        pv_target = window[self.input_len:][:, self.scalers.pv_idx]
        return {
            "x_cv": torch.from_numpy(x_cv).float(),
            "x_cv_target": torch.from_numpy(x_cv_target).float(),
            "pv_init": torch.from_numpy(pv_init).float(),
            "pv_target": torch.from_numpy(pv_target).float(),
            "scenario": torch.tensor(self.scenario, dtype=torch.long),
        }


# ── Training loop ───────────────────────────────────────────────────────────

def ss_ratio_at(epoch: int, ss_start: int, ss_end: int, ss_max: float) -> float:
    if epoch < ss_start:
        return 0.0
    if epoch >= ss_end:
        return ss_max
    return ss_max * (epoch - ss_start) / (ss_end - ss_start)


def train_one_epoch(model, loader, opt, device, ss_ratio, grad_clip) -> float:
    model.train()
    total = 0.0
    n = 0
    loss_fn = nn.MSELoss()
    for batch in loader:
        x_cv = batch["x_cv"].to(device, non_blocking=True)
        x_cv_target = batch["x_cv_target"].to(device, non_blocking=True)
        pv_init = batch["pv_init"].to(device, non_blocking=True)
        pv_target = batch["pv_target"].to(device, non_blocking=True)
        scenario = batch["scenario"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        pv_pred = model(
            x_cv, x_cv_target, pv_init, scenario,
            pv_teacher=pv_target, ss_ratio=ss_ratio,
        )
        loss = loss_fn(pv_pred, pv_target)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += loss.item() * x_cv.size(0)
        n += x_cv.size(0)
    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    loss_fn = nn.MSELoss()
    for batch in loader:
        x_cv = batch["x_cv"].to(device, non_blocking=True)
        x_cv_target = batch["x_cv_target"].to(device, non_blocking=True)
        pv_init = batch["pv_init"].to(device, non_blocking=True)
        pv_target = batch["pv_target"].to(device, non_blocking=True)
        scenario = batch["scenario"].to(device, non_blocking=True)
        pv_pred = model(x_cv, x_cv_target, pv_init, scenario)  # open loop (no teacher)
        loss = loss_fn(pv_pred, pv_target)
        total += loss.item() * x_cv.size(0)
        n += x_cv.size(0)
    return total / max(1, n)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=Path, required=True,
                    help="dir containing train*.csv / test*.csv")
    ap.add_argument("--scaler_dir", type=Path, required=True,
                    help="dir with scaler.pkl + metadata.pkl (usually generator/scalers)")
    ap.add_argument("--output_dir", type=Path, default=Path("./out_plant"))
    ap.add_argument("--init_from", type=Path, default=None,
                    help="optional existing gru_plant.pt to initialise from")
    ap.add_argument("--target_len", type=int, default=180,
                    help="rollout horizon in seconds (training decoder length)")
    ap.add_argument("--input_len", type=int, default=300)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--emb_dim", type=int, default=32)
    ap.add_argument("--n_scenarios", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ss_start", type=int, default=5)
    ap.add_argument("--ss_end", type=int, default=30)
    ap.add_argument("--ss_max", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=8,
                    help="early stopping: epochs without val improvement")
    ap.add_argument("--train_files", nargs="+",
                    default=["train1.csv", "train2.csv", "train3.csv"])
    ap.add_argument("--val_files", nargs="+", default=["train4.csv"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  target_len={args.target_len}  input_len={args.input_len}")

    # 1. Scaler
    scalers = load_scaler(args.scaler_dir)
    n_plant_in = len(scalers.plant_in_idx)
    n_pv = len(scalers.pv_idx)
    print(f"scaler OK · {len(scalers.sensor_cols)} cols · n_plant_in={n_plant_in}  n_pv={n_pv}")

    # 2. Build train + val window arrays
    def build_split(fnames: List[str], drop_attack: bool) -> Tuple[np.ndarray, np.ndarray]:
        scaled_chunks, starts_chunks = [], []
        offset = 0
        for f in fnames:
            path = args.processed_dir / f
            if not path.exists():
                print(f"  SKIP {path} (missing)")
                continue
            scaled, lbl = read_csv_scaled(path, scalers)
            starts_local = build_windows(scaled, lbl, args.input_len, args.target_len,
                                         args.stride, drop_attack=drop_attack)
            print(f"  {f}: {len(scaled):,} rows → {len(starts_local):,} windows")
            scaled_chunks.append(scaled)
            starts_chunks.append(starts_local + offset)
            offset += len(scaled)
        if not scaled_chunks:
            raise RuntimeError(f"No usable files in {fnames}")
        return np.concatenate(scaled_chunks, axis=0), np.concatenate(starts_chunks)

    print("loading train split...")
    scaled_train, starts_train = build_split(args.train_files, drop_attack=True)
    print(f"train: {len(starts_train):,} windows")
    print("loading val split...")
    scaled_val, starts_val = build_split(args.val_files, drop_attack=True)
    print(f"val  : {len(starts_val):,} windows")

    train_ds = PlantDataset(scaled_train, starts_train, scalers,
                            args.input_len, args.target_len, scenario=0)
    val_ds = PlantDataset(scaled_val, starts_val, scalers,
                          args.input_len, args.target_len, scenario=0)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 3. Model
    model = GRUPlant(
        n_plant_in=n_plant_in, n_pv=n_pv,
        hidden=args.hidden, layers=args.layers,
        n_scenarios=args.n_scenarios, emb_dim=args.emb_dim,
        n_haiend=0, dropout=args.dropout,
    ).to(device)
    if args.init_from is not None and args.init_from.exists():
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        src_state = blob["model_state"]
        # drop haiend_head keys if present in source but not in this model
        own_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in src_state.items() if k in own_keys}
        missing = own_keys - set(filtered.keys())
        unexpected = set(src_state.keys()) - set(filtered.keys())
        model.load_state_dict(filtered, strict=False)
        print(f"init from {args.init_from.name}  "
              f"(loaded {len(filtered)} / missing {len(missing)} / unexpected {len(unexpected)})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f} M params")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                      patience=3, threshold=1e-4)

    # 4. Training loop
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(args.epochs):
        ssr = ss_ratio_at(epoch, args.ss_start, args.ss_end, args.ss_max)
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, opt, device, ssr, args.grad_clip)
        vl = validate(model, val_loader, device)
        sched.step(vl)
        dur = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "val": vl, "ss": ssr, "sec": dur,
                        "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch:03d}  train {tr:.5f}  val {vl:.5f}  ss {ssr:.2f}  "
              f"lr {opt.param_groups[0]['lr']:.1e}  {dur:.0f}s")
        if vl < best_val - 1e-5:
            best_val = vl
            stale = 0
            _save_plant(model, args.output_dir / "gru_plant.pt", args, best_val)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop @ epoch {epoch}  best_val={best_val:.5f}")
                break

    with (args.output_dir / "train_history.json").open("w") as f:
        json.dump({"history": history, "best_val": best_val, "args": vars(args) |
                   {"processed_dir": str(args.processed_dir),
                    "scaler_dir": str(args.scaler_dir),
                    "output_dir": str(args.output_dir),
                    "init_from": str(args.init_from) if args.init_from else None}}, f,
                  indent=2, default=str)
    print(f"done. best_val={best_val:.5f}  saved to {args.output_dir}")


def _save_plant(model: GRUPlant, path: Path, args, val_loss: float) -> None:
    blob = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "hidden": args.hidden,
        "layers": args.layers,
        "n_haiend": model.n_haiend,
        "n_plant_in": model.n_plant_in,
        "n_pv": model.n_pv,
        "n_scenarios": args.n_scenarios,
        "emb_dim": args.emb_dim,
        "target_len": args.target_len,
        "input_len": args.input_len,
        "val_loss": float(val_loss),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)


if __name__ == "__main__":
    main()
