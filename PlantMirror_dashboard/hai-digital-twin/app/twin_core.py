"""
twin_core.py — frozen-weights inference core for the HAI Digital Twin dashboard.

Loads the gru_scenario_haiend checkpoints (GRUPlant + 4 GRUControllers + 1
CCSequenceModel) and exposes a Replay engine that streams HAI test CSVs, builds
300s input windows, and runs one-shot plant prediction plus full closed-loop
rollouts. The HAIEND head predicts 36 internal PLC signals, enabling dual-head
anomaly scoring that matches the report evaluation (sec3_detection.py).

Nothing here depends on hai-digital-twin/03_model or 02_data_pipeline — all
model classes and column lists are reverse-engineered from checkpoint metadata
and scaler metadata so the app is self-contained.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: F401

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── Constants locked from metadata.pkl / checkpoint headers ──────────────────

INPUT_LEN = 300
TARGET_LEN = 180
STRIDE = 60

PV_COLS: List[str] = ["P1_PIT01", "P1_LIT01", "P1_FT03Z", "P1_TIT01", "P1_TIT03"]

# Per-loop [SP, PV, CV] base columns and the 3 appended causal features.
# Values come from ctrl_scaler_<loop>.pkl ("cols") and checkpoint
# ["causal_layers"] respectively. Final controller input width = 6.
LOOP_SPECS: Dict[str, Dict[str, object]] = {
    "PC": {
        "base_cols": ["x1001_05_SETPOINT_OUT", "P1_PIT01", "P1_PCV01D"],
        "causal_cols": ["P1_PCV02D", "P1_FT01", "P1_TIT01"],
        "hidden": 64,
        "arch": "GRUController",
        "cv_col": "P1_PCV01D",
    },
    "LC": {
        "base_cols": ["x1002_07_SETPOINT_OUT", "P1_LIT01", "P1_LCV01D"],
        "causal_cols": ["P1_FT03", "P1_FCV03D", "P1_PCV01D"],
        "hidden": 64,
        "arch": "GRUController",
        "cv_col": "P1_LCV01D",
    },
    "FC": {
        "base_cols": ["x1002_08_SETPOINT_OUT", "P1_FT03Z", "P1_FCV03D"],
        "causal_cols": ["P1_PIT01", "P1_LIT01", "P1_TIT03"],
        "hidden": 128,
        "arch": "GRUController",
        "cv_col": "P1_FCV03D",
    },
    "TC": {
        "base_cols": ["x1003_18_SETPOINT_OUT", "P1_TIT01", "P1_FCV01D"],
        "causal_cols": ["P1_FT02", "P1_PIT02", "P1_TIT02"],
        "hidden": 64,
        "arch": "GRUController",
        "cv_col": "P1_FCV01D",
    },
    "CC": {
        "base_cols": ["P1_PP04SP", "P1_TIT03", "P1_PP04"],
        "causal_cols": ["P1_PP04D", "P1_FCV03D", "P1_PCV02D"],
        "hidden": 64,
        "arch": "CCSequenceModel",
        "cv_col": "P1_PP04",
    },
}

LOOP_ORDER: List[str] = ["PC", "LC", "FC", "TC", "CC"]

# Internal PLC function-block channels used by the HAIEND head
# (matches config.py HAIEND_COLS from the main repo)
HAIEND_COLS: List[str] = [
    '1001.13-OUT', '1001.14-OUT', '1001.15-OUT',
    '1001.16-OUT', '1001.17-OUT', '1001.20-OUT',
    '1002.9-OUT', '1002.20-OUT', '1002.21-OUT', '1002.30-OUT', '1002.31-OUT',
    '1003.5-OUT', '1003.10-OUT', '1003.11-OUT', '1003.17-OUT',
    '1003.23-OUT', '1003.24-OUT', '1003.25-OUT', '1003.26-OUT',
    '1003.29-OUT', '1003.30-OUT',
    '1020.13-OUT', '1020.14-OUT', '1020.15-OUT',
    '1020.18-OUT', '1020.20-OUT',
    'DM-PP04-D', 'DM-PP04-AO',
    'DM-TWIT-04', 'DM-TWIT-05',
    'DM-AIT-DO', 'DM-AIT-PH',
    'GATEOPEN',
    'DM-FT01Z', 'DM-FT02Z', 'DM-FT03Z',
]

SCENARIO_MAPPING = {
    0: "normal",
    1: "AP_no_combination",
    2: "AP_with_combination",
    3: "AE_no_combination",
}
SCENARIO_LABELS = list(SCENARIO_MAPPING.values())
N_SCENARIOS = 4

# Anomaly threshold (PV MSE) — same for both gru_scenario_weighted and haiend since PV weights are frozen
DEFAULT_THRESHOLD = 0.32615


# ── Model definitions (match checkpoint shapes exactly) ──────────────────────

class GRUPlant(nn.Module):
    """MIMO GRU plant: encoder over x_cv + scenario emb → decoder autoregresses PV.

    When n_haiend > 0 a second output head predicts internal PLC function-block
    signals (HAIEND channels). These expose AE attack footprints invisible in
    PV residuals alone, matching the gru_scenario_haiend evaluation setup.
    """

    def __init__(
        self,
        n_plant_in: int = 128,
        n_pv: int = 5,
        hidden: int = 512,
        layers: int = 2,
        n_scenarios: int = 4,
        emb_dim: int = 32,
        n_haiend: int = 0,
    ):
        super().__init__()
        self.n_plant_in = n_plant_in
        self.n_pv = n_pv
        self.n_haiend = n_haiend
        self.scenario_emb = nn.Embedding(n_scenarios, emb_dim)
        self.encoder = nn.GRU(n_plant_in + emb_dim, hidden, layers, batch_first=True)
        self.decoder = nn.GRU(n_plant_in + n_pv, hidden, layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_pv),
        )
        if n_haiend > 0:
            self.haiend_head = nn.Sequential(
                nn.Linear(hidden, 128),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(128, n_haiend),
            )
        else:
            self.haiend_head = None

    @torch.no_grad()
    def predict(
        self,
        x_cv: torch.Tensor,         # (B, input_len, n_plant_in)
        x_cv_target: torch.Tensor,  # (B, target_len, n_plant_in)
        pv_init: torch.Tensor,      # (B, n_pv)
        scenario: torch.Tensor,     # (B,) long
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (pv_out, haiend_out). haiend_out is None when n_haiend == 0."""
        self.eval()
        B, T, _ = x_cv_target.shape
        emb = self.scenario_emb(scenario).unsqueeze(1).expand(-1, x_cv.size(1), -1)
        _, h = self.encoder(torch.cat([x_cv, emb], dim=-1))
        pv = pv_init
        pv_outs: List[torch.Tensor] = []
        haiend_outs: List[torch.Tensor] = []
        for t in range(T):
            dec_in = torch.cat([x_cv_target[:, t, :], pv], dim=-1).unsqueeze(1)
            out, h = self.decoder(dec_in, h)
            h_out = out.squeeze(1)
            pv = self.fc(h_out)
            pv_outs.append(pv)
            if self.haiend_head is not None:
                haiend_outs.append(self.haiend_head(h_out))
        pv_tensor = torch.stack(pv_outs, dim=1)
        haiend_tensor = torch.stack(haiend_outs, dim=1) if haiend_outs else None
        return pv_tensor, haiend_tensor

    # ── Live twin primitives ──────────────────────────────────────────────

    @torch.no_grad()
    def encode_only(self, x_cv: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        """Run the encoder over a 300s warm-up window. Returns h: (L, B, hidden)."""
        self.eval()
        emb = self.scenario_emb(scenario).unsqueeze(1).expand(-1, x_cv.size(1), -1)
        _, h = self.encoder(torch.cat([x_cv, emb], dim=-1))
        return h

    @torch.no_grad()
    def step_once(
        self,
        x_cv_t: torch.Tensor,  # (B, n_plant_in)
        h: torch.Tensor,        # (L, B, hidden)
        pv_prev: torch.Tensor,  # (B, n_pv)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Advance one second. Returns (pv_new, haiend_new, h_new).
        haiend_new is None when n_haiend == 0."""
        self.eval()
        dec_in = torch.cat([x_cv_t, pv_prev], dim=-1).unsqueeze(1)
        out, h_new = self.decoder(dec_in, h)
        h_out = out.squeeze(1)
        pv_new = self.fc(h_out)
        haiend_new = self.haiend_head(h_out) if self.haiend_head is not None else None
        return pv_new, haiend_new, h_new


class GRUController(nn.Module):
    """Per-loop seq2seq: encoder over [SP, PV, CV, causal×3] → decoder autoregresses CV."""

    def __init__(self, n_inputs: int = 6, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.GRU(n_inputs, hidden, layers, batch_first=True)
        self.decoder = nn.GRU(1, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, target_len: int = TARGET_LEN) -> torch.Tensor:
        self.eval()
        B = x.size(0)
        _, h = self.encoder(x)
        prev = torch.zeros(B, 1, 1, device=x.device)
        outs: List[torch.Tensor] = []
        for _ in range(target_len):
            step, h = self.decoder(prev, h)
            cv = self.out(step)
            outs.append(cv)
            prev = cv
        return torch.cat(outs, dim=1)  # (B, target_len, 1)


class CCSequenceModel(nn.Module):
    """CC loop: same encoder/decoder as GRUController, two heads (on/cv)."""

    def __init__(self, n_inputs: int = 6, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.GRU(n_inputs, hidden, layers, batch_first=True)
        self.decoder = nn.GRU(1, hidden, layers, batch_first=True)
        self.head_on = nn.Linear(hidden, 1)
        self.head_cv = nn.Linear(hidden, 1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, target_len: int = TARGET_LEN,
                on_threshold: float = 0.5) -> torch.Tensor:
        """Returns gated CV: 0 when pump is off, head_cv when on."""
        self.eval()
        B = x.size(0)
        _, h = self.encoder(x)
        prev = torch.zeros(B, 1, 1, device=x.device)
        outs: List[torch.Tensor] = []
        for _ in range(target_len):
            step, h = self.decoder(prev, h)
            cv = self.head_cv(step)
            logit = self.head_on(step)
            gated = (torch.sigmoid(logit) > on_threshold).float() * cv
            outs.append(gated)
            prev = gated
        return torch.cat(outs, dim=1)


# ── Scaler / column wrapper ──────────────────────────────────────────────────

@dataclass
class TwinScalers:
    """Holds the plant scaler (133 cols) and the 5 per-loop ctrl scalers."""

    plant_mean: np.ndarray
    plant_scale: np.ndarray
    sensor_cols: List[str]
    plant_in_cols: List[str]    # 128 cols = sensor_cols minus PVs
    plant_in_idx: np.ndarray    # indices of plant_in_cols inside sensor_cols
    pv_idx: np.ndarray          # indices of PV_COLS inside sensor_cols
    ctrl: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def scale_plant_row(self, df: pd.DataFrame) -> np.ndarray:
        """Return (T, 133) scaled array using plant_scaler for all sensor_cols."""
        arr = df[self.sensor_cols].to_numpy(dtype=np.float32)
        return (arr - self.plant_mean) / self.plant_scale

    def inverse_plant(self, arr_scaled: np.ndarray, cols: List[str]) -> np.ndarray:
        """Inverse-transform a slice of scaled values back to physical units."""
        idx = np.array([self.sensor_cols.index(c) for c in cols], dtype=np.int64)
        return arr_scaled * self.plant_scale[idx] + self.plant_mean[idx]

    def scale_ctrl(self, loop: str, base_df: pd.DataFrame) -> np.ndarray:
        """Scale the 3 base ctrl cols with the loop's StandardScaler."""
        cs = self.ctrl[loop]
        cols: List[str] = cs["cols"]  # type: ignore[assignment]
        mean: np.ndarray = cs["mean"]  # type: ignore[assignment]
        scale: np.ndarray = cs["scale"]  # type: ignore[assignment]
        arr = base_df[cols].to_numpy(dtype=np.float32)
        return (arr - mean) / scale

    def inverse_ctrl_cv(self, loop: str, cv_scaled: np.ndarray) -> np.ndarray:
        """Inverse-scale the CV channel of a loop (3rd col of its ctrl scaler)."""
        cs = self.ctrl[loop]
        mean: np.ndarray = cs["mean"]  # type: ignore[assignment]
        scale: np.ndarray = cs["scale"]  # type: ignore[assignment]
        return cv_scaled * scale[2] + mean[2]


def load_scalers(split_dir: Path) -> TwinScalers:
    plant = joblib.load(split_dir / "scaler.pkl")
    with open(split_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    sensor_cols: List[str] = list(meta["sensor_cols"])
    pv_idx = np.array([sensor_cols.index(p) for p in PV_COLS], dtype=np.int64)
    plant_in_cols = [c for c in sensor_cols if c not in PV_COLS]
    plant_in_idx = np.array(
        [sensor_cols.index(c) for c in plant_in_cols], dtype=np.int64
    )

    ctrl: Dict[str, Dict[str, object]] = {}
    for loop in LOOP_ORDER:
        obj = joblib.load(split_dir / f"ctrl_scaler_{loop}.pkl")
        ctrl[loop] = {
            "cols": list(obj["cols"]),
            "mean": obj["scaler"].mean_.astype(np.float32),
            "scale": obj["scaler"].scale_.astype(np.float32),
        }

    return TwinScalers(
        plant_mean=plant.mean_.astype(np.float32),
        plant_scale=plant.scale_.astype(np.float32),
        sensor_cols=sensor_cols,
        plant_in_cols=plant_in_cols,
        plant_in_idx=plant_in_idx,
        pv_idx=pv_idx,
        ctrl=ctrl,
    )


# ── Model bundle ─────────────────────────────────────────────────────────────

@dataclass
class TwinBundle:
    plant: GRUPlant
    controllers: Dict[str, nn.Module]
    scalers: TwinScalers
    device: torch.device
    threshold: float = DEFAULT_THRESHOLD
    ctrl_cv_col_idx: Dict[str, int] = field(default_factory=dict)
    pv_in_plant_in: np.ndarray = field(default_factory=lambda: np.zeros(0))
    haiend_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def to_device(self):
        self.plant.to(self.device).eval()
        for m in self.controllers.values():
            m.to(self.device).eval()
        return self


def load_bundle(
    ckpt_dir: Path,
    split_dir: Path,
    device: Optional[torch.device] = None,
) -> TwinBundle:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scalers = load_scalers(split_dir)

    # Plant — derive architecture from model_state when top-level keys are absent
    # (gru_scenario_haiend only stores hidden/layers/n_haiend/val_loss at top level)
    d = torch.load(ckpt_dir / "gru_plant.pt", map_location="cpu", weights_only=False)
    ms = d["model_state"]
    emb_dim    = ms["scenario_emb.weight"].shape[1]
    n_plant_in = ms["encoder.weight_ih_l0"].shape[1] - emb_dim
    n_pv       = ms["fc.3.weight"].shape[0]
    n_scenarios = ms["scenario_emb.weight"].shape[0]
    n_haiend   = ms["haiend_head.3.weight"].shape[0] if "haiend_head.3.weight" in ms else 0
    plant = GRUPlant(
        n_plant_in=d.get("n_plant_in", n_plant_in),
        n_pv=d.get("n_pv", n_pv),
        hidden=d["hidden"],
        layers=d["layers"],
        n_scenarios=d.get("n_scenarios", n_scenarios),
        n_haiend=n_haiend,
    )
    plant.load_state_dict(ms, strict=False)

    # Controllers
    controllers: Dict[str, nn.Module] = {}
    for loop in LOOP_ORDER:
        d = torch.load(ckpt_dir / f"gru_ctrl_{loop.lower()}.pt",
                       map_location="cpu", weights_only=False)
        if d["arch"] == "CCSequenceModel":
            m: nn.Module = CCSequenceModel(
                n_inputs=d["n_inputs"], hidden=d["hidden"], layers=d["layers"]
            )
        else:
            m = GRUController(
                n_inputs=d["n_inputs"], hidden=d["hidden"], layers=d["layers"]
            )
        m.load_state_dict(d["model_state"])
        controllers[loop] = m

    # Indexing helpers
    ctrl_cv_col_idx = {
        loop: scalers.plant_in_cols.index(spec["cv_col"])  # type: ignore[arg-type]
        for loop, spec in LOOP_SPECS.items()
    }

    # HAIEND column indices in the full 133-column scaled sensor matrix
    haiend_idx = np.array(
        [scalers.sensor_cols.index(c) for c in HAIEND_COLS
         if c in scalers.sensor_cols],
        dtype=np.int64,
    )

    return TwinBundle(
        plant=plant,
        controllers=controllers,
        scalers=scalers,
        device=device,
        ctrl_cv_col_idx=ctrl_cv_col_idx,
        haiend_idx=haiend_idx,
    ).to_device()


# ── Replay engine ────────────────────────────────────────────────────────────

@dataclass
class ReplaySource:
    """One HAI test CSV loaded into memory in scaled form."""

    name: str
    df_raw: pd.DataFrame              # full CSV (raw units, 1 Hz)
    scaled: np.ndarray                # (T, 133) scaled sensor matrix
    timestamps: pd.Series
    attack_meta: pd.DataFrame         # attack_id, scenario, attack_type, ...

    def __len__(self) -> int:
        return len(self.df_raw)


def load_replay(csv_path: Path, scalers: TwinScalers) -> ReplaySource:
    df = pd.read_csv(csv_path)
    # Some raw dumps have extra whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    scaled = scalers.scale_plant_row(df)
    meta_cols = [c for c in [
        "attack_id", "scenario", "attack_type", "combination",
        "target_controller", "target_points", "label"
    ] if c in df.columns]
    attack_meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.Series(
        pd.RangeIndex(len(df))
    )
    return ReplaySource(
        name=csv_path.stem, df_raw=df, scaled=scaled, timestamps=ts,
        attack_meta=attack_meta,
    )


# ── Window builders ──────────────────────────────────────────────────────────

def _causal_stack(src: ReplaySource, scalers: TwinScalers, cols: List[str],
                  start: int, length: int) -> np.ndarray:
    """Fetch scaled columns from the plant-scaled matrix for a window."""
    idx = np.array([scalers.sensor_cols.index(c) for c in cols], dtype=np.int64)
    return src.scaled[start:start + length, idx]  # (length, len(cols))


def build_plant_window(
    bundle: TwinBundle, src: ReplaySource, t_end: int
) -> Optional[Dict[str, torch.Tensor]]:
    """Build plant input for a window ending at t_end (exclusive of target span).

    Requires t_end - INPUT_LEN >= 0 and t_end + TARGET_LEN <= len(src). Returns
    tensors on the bundle device, or None if window is out of bounds.
    """
    t0 = t_end - INPUT_LEN
    t1 = t_end
    t2 = t_end + TARGET_LEN
    if t0 < 0 or t2 > len(src):
        return None
    s = bundle.scalers
    full_in = src.scaled[t0:t1]              # (300, 133)
    full_tgt = src.scaled[t1:t2]             # (180, 133)

    x_cv = full_in[:, s.plant_in_idx]        # (300, 128)
    x_cv_target = full_tgt[:, s.plant_in_idx]  # (180, 128)
    pv_init = full_in[-1, s.pv_idx]          # (5,)
    pv_target = full_tgt[:, s.pv_idx]        # (180, 5)

    scen = 0
    if "scenario" in src.df_raw.columns:
        # scenario column uses strings like "AP01" / "normal" in processed CSVs.
        # Fall back to 0 (normal) unless label=1 at any point in the window.
        win_meta = src.df_raw.iloc[t0:t1]
        if "label" in win_meta.columns and (win_meta["label"] > 0).any():
            at = win_meta.loc[win_meta["label"] > 0, "attack_type"].dropna()
            combo = win_meta.loc[win_meta["label"] > 0, "combination"].dropna()
            if len(at):
                at0, c0 = at.iloc[0], combo.iloc[0] if len(combo) else "no"
                if at0 == "actuator_pollution" and c0 == "no":
                    scen = 1
                elif at0 == "actuator_pollution" and c0 == "with":
                    scen = 2
                elif at0 == "actuator_emulation":
                    scen = 3

    device = bundle.device
    return {
        "x_cv": torch.from_numpy(x_cv).unsqueeze(0).to(device).float(),
        "x_cv_target": torch.from_numpy(x_cv_target).unsqueeze(0).to(device).float(),
        "pv_init": torch.from_numpy(pv_init).unsqueeze(0).to(device).float(),
        "pv_target": torch.from_numpy(pv_target).unsqueeze(0).to(device).float(),
        "scenario": torch.tensor([scen], dtype=torch.long, device=device),
        "t0": torch.tensor([t0]), "t1": torch.tensor([t1]), "t2": torch.tensor([t2]),
    }


def build_ctrl_window(
    bundle: TwinBundle, src: ReplaySource, t_end: int, loop: str
) -> Optional[torch.Tensor]:
    """Build a controller input of shape (1, 300, 6)."""
    t0 = t_end - INPUT_LEN
    if t0 < 0 or t_end > len(src):
        return None
    spec = LOOP_SPECS[loop]
    base_df = src.df_raw.iloc[t0:t_end]
    base_scaled = bundle.scalers.scale_ctrl(loop, base_df)  # (300, 3)
    causal = _causal_stack(
        src, bundle.scalers, spec["causal_cols"], t0, INPUT_LEN  # type: ignore[arg-type]
    )  # (300, 3)
    full = np.concatenate([base_scaled, causal], axis=1).astype(np.float32)  # (300, 6)
    return torch.from_numpy(full).unsqueeze(0).to(bundle.device)


# ── Inference primitives ─────────────────────────────────────────────────────

def predict_plant(bundle: TwinBundle, win: Dict[str, torch.Tensor]) -> np.ndarray:
    """One-shot plant rollout given a replay window. Returns scaled PV (180, 5)."""
    pv, _haiend = bundle.plant.predict(
        win["x_cv"], win["x_cv_target"], win["pv_init"], win["scenario"]
    )
    return pv.squeeze(0).cpu().numpy()


def window_mse(pv_pred_scaled: np.ndarray, pv_true_scaled: np.ndarray) -> float:
    """Window-level MSE in scaled space (matches training-time detector)."""
    return float(np.mean((pv_pred_scaled - pv_true_scaled) ** 2))


def per_step_residual(pv_pred_scaled: np.ndarray,
                      pv_true_scaled: np.ndarray) -> np.ndarray:
    """Per-timestep squared-error, mean across PVs. Shape (target_len,)."""
    return np.mean((pv_pred_scaled - pv_true_scaled) ** 2, axis=1)


def per_pv_mse(pv_pred_scaled: np.ndarray,
               pv_true_scaled: np.ndarray) -> np.ndarray:
    """MSE per PV channel. Shape (n_pv,)."""
    return np.mean((pv_pred_scaled - pv_true_scaled) ** 2, axis=0)


# ── Closed-loop rollout (Generative twin) ────────────────────────────────────

def closed_loop_rollout(
    bundle: TwinBundle,
    src: ReplaySource,
    t_end: int,
    sp_overrides: Optional[Dict[str, float]] = None,
    scenario: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Run all 5 controllers, splice CVs into x_cv_target, then run the plant.

    sp_overrides: {loop: new_sp_raw_value} to replace the replay's SP column for
    the 300s history window (what-if sliders).
    """
    t0 = t_end - INPUT_LEN
    t2 = t_end + TARGET_LEN
    if t0 < 0 or t2 > len(src):
        return None

    # Clone the raw slice so SP overrides don't leak into src.df_raw
    raw = src.df_raw.iloc[t0:t2].copy()
    if sp_overrides:
        for loop, sp_val in sp_overrides.items():
            sp_col = LOOP_SPECS[loop]["base_cols"][0]  # type: ignore[index]
            raw[sp_col] = float(sp_val)

    # Re-scale the plant-side view with the overridden SPs baked in
    scaled_local = bundle.scalers.scale_plant_row(raw)  # (480, 133)
    x_cv_in = scaled_local[:INPUT_LEN, bundle.scalers.plant_in_idx]
    x_cv_tgt = scaled_local[INPUT_LEN:, bundle.scalers.plant_in_idx].copy()
    pv_init = scaled_local[INPUT_LEN - 1, bundle.scalers.pv_idx]

    ctrl_cv_preds: Dict[str, np.ndarray] = {}
    for loop in LOOP_ORDER:
        spec = LOOP_SPECS[loop]
        # Base ctrl cols scaled by ctrl_scaler_<loop>
        base_df = raw.iloc[:INPUT_LEN]
        base_scaled = bundle.scalers.scale_ctrl(loop, base_df)
        causal = scaled_local[
            :INPUT_LEN,
            np.array([bundle.scalers.sensor_cols.index(c)
                      for c in spec["causal_cols"]],  # type: ignore[arg-type]
                     dtype=np.int64),
        ]
        ctrl_in = np.concatenate([base_scaled, causal], axis=1).astype(np.float32)
        ctrl_in_t = torch.from_numpy(ctrl_in).unsqueeze(0).to(bundle.device)
        cv_scaled = bundle.controllers[loop].predict(
            ctrl_in_t, target_len=TARGET_LEN
        ).squeeze(0).squeeze(-1).cpu().numpy()  # (180,)
        ctrl_cv_preds[loop] = cv_scaled

        # Splice CV into x_cv_target at the plant-side column for this CV
        cv_col = spec["cv_col"]  # type: ignore[assignment]
        cv_sensor_idx = bundle.scalers.sensor_cols.index(cv_col)
        cv_plant_idx = int(np.where(bundle.scalers.plant_in_idx == cv_sensor_idx)[0][0])
        # Convert ctrl-scaled CV → physical → plant-scaled
        cv_physical = bundle.scalers.inverse_ctrl_cv(loop, cv_scaled)
        plant_mean = bundle.scalers.plant_mean[cv_sensor_idx]
        plant_scale = bundle.scalers.plant_scale[cv_sensor_idx]
        x_cv_tgt[:, cv_plant_idx] = (cv_physical - plant_mean) / plant_scale

    # Scenario override
    if scenario is None:
        scenario_val = 0
    else:
        scenario_val = int(scenario)

    x_cv_in_t = torch.from_numpy(x_cv_in).unsqueeze(0).to(bundle.device).float()
    x_cv_tgt_t = torch.from_numpy(x_cv_tgt).unsqueeze(0).to(bundle.device).float()
    pv_init_t = torch.from_numpy(pv_init).unsqueeze(0).to(bundle.device).float()
    scen_t = torch.tensor([scenario_val], dtype=torch.long, device=bundle.device)

    pv_tensor, _haiend = bundle.plant.predict(
        x_cv_in_t, x_cv_tgt_t, pv_init_t, scen_t
    )
    pv_scaled = pv_tensor.squeeze(0).cpu().numpy()  # (180, 5)

    pv_physical = bundle.scalers.inverse_plant(pv_scaled, PV_COLS)

    return {
        "pv_scaled": pv_scaled,
        "pv_physical": pv_physical,
        "ctrl_cv_scaled": ctrl_cv_preds,
        "x_cv_target_used": x_cv_tgt,
        "scenario": scenario_val,
    }


# ── Convenience ──────────────────────────────────────────────────────────────

def default_paths() -> Dict[str, Path]:
    """Resolve default checkpoint + split-dir paths relative to this file."""
    here = Path(__file__).resolve().parent
    # app/ → hai-digital-twin/ (dashboard) → PlantMirror_dashboard/ → hai-digital-twin/ (main repo)
    repo = here.parent.parent.parent
    return {
        "ckpt_dir": repo / "outputs" / "pipeline" / "gru_scenario_haiend",
        "split_dir": repo / "outputs" / "scaled_split",
        "causal_json": repo / "outputs" / "causal_graph" / "parents_full.json",
        "test_csvs": [
            repo / "00_data" / "processed" / "test1.csv",
            repo / "00_data" / "processed" / "test2.csv",
        ],
    }


if __name__ == "__main__":
    # Smoke test: load everything and run one prediction.
    paths = default_paths()
    print(f"Loading bundle from {paths['ckpt_dir']} ...")
    bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
    print(f"  device={bundle.device}  threshold={bundle.threshold}")
    print(f"  plant n_plant_in={bundle.plant.n_plant_in}  n_pv={bundle.plant.n_pv}")
    print(f"  controllers: {list(bundle.controllers.keys())}")

    src = load_replay(paths["test_csvs"][0], bundle.scalers)
    print(f"Loaded replay '{src.name}' ({len(src)} rows)")

    t_end = INPUT_LEN
    win = build_plant_window(bundle, src, t_end)
    assert win is not None
    pv_pred = predict_plant(bundle, win)
    pv_true = win["pv_target"].squeeze(0).cpu().numpy()
    mse = window_mse(pv_pred, pv_true)
    print(f"  t_end={t_end}  window MSE={mse:.5f}  "
          f"(threshold={bundle.threshold})  -> "
          f"{'ANOMALY' if mse > bundle.threshold else 'normal'}")

    rollout = closed_loop_rollout(bundle, src, t_end, scenario=0)
    assert rollout is not None
    print(f"  closed-loop pv_physical shape: {rollout['pv_physical'].shape}")
    print("OK")
