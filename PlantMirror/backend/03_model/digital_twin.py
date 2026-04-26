"""
digital_twin.py — GRU Digital Twin: closed-loop simulation + residual anomaly detection.

Architecture:
    [GRU Controllers x4] -> CVs -> [GRU Plant] -> PV_predicted
    Residual = PV_actual - PV_predicted
    Score    = MSE(PV_pred, PV_actual)  per window
    Attack   = score > threshold

Supports two checkpoint styles:
  - Standard (n_inputs=2): [SP, PV]
  - Causal   (n_inputs=6): [SP, PV, CV_fb, causal1, causal2, causal3]
    Controller inputs are built from raw full-sensor arrays.

Usage:
    from digital_twin import DigitalTwin
    twin = DigitalTwin("outputs/gru_scenario_weighted/gru_scenario_weighted", device="cpu", data=data)
    raw_val  = np.load("outputs/scaled_split/val_data.npz")["X"]
    raw_test = np.load("outputs/scaled_split/test_data.npz")["X"]
    twin.calibrate(data, raw_val=raw_val)
    results = twin.run_batch(X_test, X_cv_tgt, pv_init, scenario, pv_actual,
                             ctrl_data=twin.build_ctrl_inputs(raw_test))
"""

import json
import sys
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from gru import GRUPlant, GRUController
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR

CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC', 'CC']


# ── CCSequenceModel ───────────────────────────────────────────────────────────

class CCSequenceModel(nn.Module):
    """
    CC cooling loop model from teammate's GRU-Scenario-Weighted checkpoint.

    Same GRU encoder/decoder as GRUController but two output heads:
        head_on  : sigmoid -> pump on/off probability
        head_cv  : pump speed

    predict() returns (B, target_len, 1) CV sequence — same interface as GRUController.
    """

    def __init__(self, n_inputs: int = 6, hidden: int = 64,
                 layers: int = 2, output_len: int = 180):
        super().__init__()
        self.output_len = output_len
        drop = 0.1 if layers > 1 else 0.0

        self.encoder = nn.GRU(n_inputs, hidden, layers,
                              batch_first=True, dropout=drop)
        self.decoder = nn.GRU(1, hidden, layers,
                              batch_first=True, dropout=drop)
        self.head_on = nn.Linear(hidden, 1)
        self.head_cv = nn.Linear(hidden, 1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                target_len: Optional[int] = None) -> torch.Tensor:
        """
        x: (B, input_len, n_inputs)
        Returns (B, target_len, 1) pump CV (0 when pump predicted off).
        """
        self.eval()
        T = target_len if target_len is not None else self.output_len
        B = x.size(0)
        _, h = self.encoder(x)
        prev = torch.zeros(B, 1, 1, device=x.device)
        outputs = []
        for _ in range(T):
            out, h = self.decoder(prev, h)
            pump_on = torch.sigmoid(self.head_on(out)) > 0.5
            speed   = self.head_cv(out)
            cv      = pump_on.float() * speed
            outputs.append(cv)
            prev = cv
        return torch.cat(outputs, dim=1)


# ── DigitalTwin ───────────────────────────────────────────────────────────────

class DigitalTwin:
    """
    GRU-based Digital Twin for the HAI industrial process.

    Wraps pre-trained GRU plant + controllers (standard or causal 6-input).
    Provides closed-loop simulation and residual-based attack detection.
    """

    def __init__(self, gru_dir: str | Path, device: str = "cpu", data: dict = None):
        """
        gru_dir : checkpoint directory (contains *.pt + results.json)
        device  : "cpu" or "cuda"
        data    : result of load_and_prepare_data()
        """
        self.device  = torch.device(device)
        self.gru_dir = Path(gru_dir)

        if data is None:
            raise ValueError("Pass data=load_and_prepare_data() — needed for shape info")

        self.target_len  = data['metadata']['target_len']
        self.sensor_cols = data['metadata']['sensor_cols']
        n_plant_in       = data['plant']['n_plant_in']
        n_pv             = data['plant']['n_pv']
        n_scenarios      = data['metadata']['n_scenarios']
        self.n_pv        = n_pv

        # Non-PV column mapping (for patching CV columns in plant decoder)
        pv_set      = set(PV_COLS)
        non_pv_cols = [c for c in self.sensor_cols if c not in pv_set]
        col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
        self.ctrl_cv_col_idx = {
            ln: col_to_idx[LOOPS[ln].cv]
            for ln in CTRL_LOOPS
            if LOOPS[ln].cv in col_to_idx
        }

        # Load GRU plant
        ckpt = torch.load(self.gru_dir / "gru_plant.pt",
                          map_location=self.device, weights_only=False)
        self.plant = GRUPlant(
            n_plant_in  = n_plant_in,
            n_pv        = n_pv,
            hidden      = ckpt["hidden"],
            layers      = ckpt["layers"],
            n_scenarios = n_scenarios,
        ).to(self.device)
        self.plant.load_state_dict(ckpt["model_state"])
        self.plant.eval()
        self._plant_epoch    = ckpt.get("epoch", "?")
        self._plant_val_loss = ckpt.get("val_loss", float("nan"))

        # Detect causal mode from results.json
        results_path = self.gru_dir / "results.json"
        self.causal_features = {}
        if results_path.exists():
            with open(results_path) as f:
                res = json.load(f)
            self.causal_features = res.get("causal_layers", {})

        # Build column index for causal extra features (double-normalised at inference)
        # Indexed into FULL sensor_cols (all columns including PVs)
        raw_col_idx = {c: i for i, c in enumerate(self.sensor_cols)}
        self.causal_col_idx: dict = {}   # loop -> [cvfb_idx, c1, c2, c3] in sensor_cols
        for ln in CTRL_LOOPS:
            if ln not in self.causal_features:
                continue
            loop   = LOOPS[ln]
            cvfb_name = loop.cv_fb if loop.cv_fb else loop.cv
            cvfb_i    = raw_col_idx.get(cvfb_name)
            causals   = [raw_col_idx.get(c) for c in self.causal_features[ln]]
            if cvfb_i is None or None in causals:
                print(f"  WARNING: missing causal cols for {ln}, skipping")
                continue
            self.causal_col_idx[ln] = [cvfb_i] + causals

        self.causal_mode = bool(self.causal_col_idx)

        # Load plant scaler and ctrl scalers (needed to build 6-input ctrl arrays)
        self._plant_scaler  = None
        self._ctrl_scalers  = {}
        data_dir = Path(PROCESSED_DATA_DIR)
        scaler_path = data_dir / "scaler.pkl"
        if scaler_path.exists():
            self._plant_scaler = joblib.load(str(scaler_path))
        for ln in CTRL_LOOPS:
            cs_path = data_dir / f"ctrl_scaler_{ln}.pkl"
            if cs_path.exists():
                self._ctrl_scalers[ln] = joblib.load(str(cs_path))['scaler']

        # Load controllers
        self.ctrls: dict = {}
        for ln in CTRL_LOOPS:
            path = self.gru_dir / f"gru_ctrl_{ln.lower()}.pt"
            if not path.exists():
                print(f"  WARNING: {path.name} not found, skipping {ln}")
                continue
            c    = torch.load(path, map_location=self.device, weights_only=False)
            arch = c.get("arch", "GRUController")

            if arch == "CCSequenceModel":
                m = CCSequenceModel(
                    n_inputs   = c["n_inputs"],
                    hidden     = c["hidden"],
                    layers     = c["layers"],
                    output_len = self.target_len,
                ).to(self.device)
            else:
                m = GRUController(
                    n_inputs   = c["n_inputs"],
                    hidden     = c["hidden"],
                    layers     = c["layers"],
                    output_len = self.target_len,
                ).to(self.device)
            m.load_state_dict(c["model_state"])
            m.eval()
            self.ctrls[ln] = m

        self.threshold     = None
        self.threshold_fpr = None

        print(f"DigitalTwin ready | device={device}")
        print(f"  Plant  : epoch={self._plant_epoch}, val_loss={self._plant_val_loss:.6f}")
        print(f"  Ctrls  : {list(self.ctrls.keys())}")
        print(f"  Mode   : {'causal (6-input)' if self.causal_mode else 'standard (2-input)'}")
        print(f"  CV map : {self.ctrl_cv_col_idx}")
        print(f"  Threshold = NOT SET (call calibrate() first)")

    # ── Causal input builder ──────────────────────────────────────────────────

    def build_ctrl_inputs(self, raw_X: np.ndarray, ctrl_data: dict) -> dict:
        """
        Build 6-input controller arrays matching the teammate's training format:
            [SP_ctrl, PV_ctrl, CV_fb_ctrl, causal1_dn, causal2_dn, causal3_dn]

        Where:
          SP_ctrl, PV_ctrl = controller-scaled (from pipeline ctrl_data, 2 cols)
          CV_fb_ctrl       = CV feedback inverse-transformed to physical units,
                             then rescaled by ctrl scaler col [2] (CV normalisation)
          causalN_dn       = double-normalised: (plant_scaled_value - mean) / scale
                             (the training augment_ctrl_data applied scaler twice)

        Parameters
        ----------
        raw_X     : (N, seq_len, n_sensor_cols) full-sensor plant-scaled array
                    Load: np.load('outputs/scaled_split/test_data.npz')['X']
        ctrl_data : standard pipeline ctrl dict (has X_test with 2-col SP/PV)

        Returns
        -------
        {loop_name: {'X_test': (N, seq_len, 6)}}
        """
        if self._plant_scaler is None:
            raise RuntimeError("Plant scaler not loaded. Check PROCESSED_DATA_DIR path.")
        pm = self._plant_scaler.mean_
        ps = self._plant_scaler.scale_

        out = {}
        for ln, col_list in self.causal_col_idx.items():
            cvfb_i      = col_list[0]
            causal_idxs = col_list[1:]

            # SP, PV in controller scale (2 cols from pipeline ctrl_data)
            sp_pv = ctrl_data[ln]['X_test'][:, :, :2]  # (N, T, 2)

            # CV_fb: plant-scaled → physical → controller-scaled
            cvfb_plant = raw_X[:, :, [cvfb_i]].astype(np.float32)
            cvfb_phys  = cvfb_plant * ps[cvfb_i] + pm[cvfb_i]
            cs = self._ctrl_scalers.get(ln)
            if cs is not None and len(cs.mean_) >= 3:
                cvfb_ctrl = (cvfb_phys - cs.mean_[2]) / cs.scale_[2]
            else:
                cvfb_ctrl = (cvfb_phys - pm[cvfb_i]) / ps[cvfb_i]  # fallback: plant scale
            cvfb_ctrl = cvfb_ctrl.astype(np.float32)

            # Causal features: double-normalised (training augment applied scaler on
            # already-plant-scaled data, so we must do the same at inference)
            causal_parts = []
            for ci in causal_idxs:
                already_scaled = raw_X[:, :, [ci]].astype(np.float32)
                double_norm    = (already_scaled - pm[ci]) / ps[ci]
                causal_parts.append(double_norm)

            X6 = np.concatenate([sp_pv, cvfb_ctrl] + causal_parts, axis=-1)
            out[ln] = {'X_test': X6.astype(np.float32)}
        return out

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self, data: dict, fpr_target: float = 0.05,
                  raw_val: np.ndarray = None,
                  use_controllers: bool = True) -> float:
        """
        Set detection threshold from the VALIDATION set (no test leakage).

        Parameters
        ----------
        data            : result of load_and_prepare_data()
        fpr_target      : allowed false positive rate (default 0.05)
        raw_val         : (N_val, input_len, n_sensor_cols) full-sensor val array.
                          Required in causal mode when use_controllers=True.
        use_controllers : if False, skip controller patching (use actual CV signals).
                          Recommended for causal models — gives higher AUROC because
                          attack-affected CV signals carry the attack signature.
        """
        plant = data['plant']
        ctrl  = data['ctrl']

        X_val   = plant['X_val']
        Xct_val = plant['X_cv_target_val']
        pvi_val = plant['pv_init_val']
        sc_val  = plant['scenario_val']
        pv_val  = plant['pv_target_val']

        n_val_attacks = int(plant['attack_val'].sum())
        if n_val_attacks > 0:
            print(f"  WARNING: val set has {n_val_attacks} attack windows — "
                  "calibration may be slightly biased")

        if use_controllers:
            if self.causal_mode:
                if raw_val is None:
                    raise ValueError(
                        "raw_val required in causal mode with use_controllers=True.")
                # Remap val ctrl arrays under 'X_test' key so build_ctrl_inputs works
                ctrl_val_remapped = {ln: {'X_test': ctrl[ln]['X_val']}
                                     for ln in ctrl if 'X_val' in ctrl[ln]}
                ctrl_val = self.build_ctrl_inputs(raw_val, ctrl_val_remapped)
            else:
                ctrl_val = {ln: {'X_test': ctrl[ln]['X_val']}
                            for ln in ctrl if 'X_val' in ctrl[ln]}
        else:
            ctrl_val = None   # plant uses actual CV signals from X_cv_target

        res = self.run_batch(X_val, Xct_val, pvi_val, sc_val, pv_val,
                             ctrl_data=ctrl_val)
        val_scores = res['scores']

        percentile     = (1.0 - fpr_target) * 100.0
        self.threshold = float(np.percentile(val_scores, percentile))
        self.threshold_fpr = fpr_target

        print(f"  Calibrated threshold = {self.threshold:.6f}  "
              f"(val {percentile:.0f}th pct, FPR target={fpr_target*100:.0f}%)")
        return self.threshold

    # ── Core inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_batch(self, x_cv_b, x_cv_tgt_b, pv_init_b, sc_b, ctrl_X: dict) -> np.ndarray:
        """Single mini-batch closed-loop prediction -> (B, target_len, n_pv) numpy."""
        xct = x_cv_tgt_b.clone()
        for ln, col_idx in self.ctrl_cv_col_idx.items():
            if ln in self.ctrls and ln in ctrl_X:
                cv_pred = self.ctrls[ln].predict(ctrl_X[ln], target_len=self.target_len)
                xct[:, :, col_idx:col_idx + 1] = cv_pred
        pv_seq = self.plant.predict(x_cv_b, xct, pv_init_b, sc_b)
        return pv_seq.cpu().numpy()

    def run_batch(
        self,
        X_test:    np.ndarray,
        X_cv_tgt:  np.ndarray,
        pv_init:   np.ndarray,
        scenario:  np.ndarray,
        pv_actual: np.ndarray,
        ctrl_data: dict = None,
        batch_size: int  = 64,
    ) -> dict:
        """
        Run digital twin on a full dataset split.

        Parameters
        ----------
        X_test    : (N, input_len, n_plant_in)   numpy
        X_cv_tgt  : (N, target_len, n_plant_in)  numpy
        pv_init   : (N, n_pv)                    numpy
        scenario  : (N,)                          numpy int
        pv_actual : (N, target_len, n_pv)         numpy
        ctrl_data : {ln: {'X_test': (N, ...)}}
                    From build_ctrl_inputs() [causal] or pipeline ctrl dict [standard]
        batch_size: mini-batch size

        Returns
        -------
        dict: pv_pred, residuals, scores, is_attack
        """
        N       = len(X_test)
        pv_pred = np.zeros((N, self.target_len, self.n_pv), dtype=np.float32)

        for i in range(0, N, batch_size):
            sl        = slice(i, i + batch_size)
            x_cv_b    = torch.tensor(X_test[sl]).float().to(self.device)
            xct_b     = torch.tensor(X_cv_tgt[sl]).float().to(self.device)
            pv_init_b = torch.tensor(pv_init[sl]).float().to(self.device)
            sc_b      = torch.tensor(scenario[sl]).long().to(self.device)

            ctrl_X = {}
            if ctrl_data is not None:
                for ln in CTRL_LOOPS:
                    if ln in self.ctrls and ln in ctrl_data:
                        ctrl_X[ln] = torch.tensor(
                            ctrl_data[ln]['X_test'][sl]
                        ).float().to(self.device)

            pv_pred[i:i + x_cv_b.size(0)] = self._run_batch(
                x_cv_b, xct_b, pv_init_b, sc_b, ctrl_X)

        residuals = pv_actual - pv_pred
        scores    = np.mean(residuals ** 2, axis=(1, 2))
        is_attack = (scores > self.threshold
                     if self.threshold is not None
                     else np.zeros(len(scores), dtype=bool))

        return {
            "pv_pred":   pv_pred,
            "residuals": residuals,
            "scores":    scores,
            "is_attack": is_attack,
        }

    # ── Single-window helpers ─────────────────────────────────────────────────

    def score(self, pv_pred: np.ndarray, pv_actual: np.ndarray) -> float:
        """Anomaly score for one window. Shape: (T, n_pv) or (1, T, n_pv)."""
        return float(np.mean((pv_actual - pv_pred) ** 2))

    def detect(self, score: float) -> bool:
        """True if score indicates an attack."""
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before detect()")
        return score > self.threshold
