"""
twin_runtime.py — the *digital-twin* monitoring runtime.

Semantics: the twin is a persistent simulation that maintains its own
(h_plant, pv_twin) state across simulated seconds. Replay supplies the CV
input samples each tick, and the twin autoregresses one plant decoder step.

Key invariant (proven by test_math_identity.py): running `encode_only()` once
and `step_once()` N times is bit-identical to `plant.predict()` over the same
inputs, so stateful stepping is mathematically equivalent to the training-time
rollout — no retraining, no drift from new code.

Public surface:
    TwinRuntime(bundle, src)
        .warm_up(cursor, scenario)      # initialize h_plant + pv_twin
        .step(n=1)                      # advance n simulated seconds
        .reset(cursor=None)             # re-warm at current or given cursor
        .snapshot() -> TwinSnapshot     # deep-copy of state for what-if tabs
        .rolling_arrays()               # current buffers for plotting

    TwinSnapshot — frozen state passed to Predictive / Generative tabs so the
    live runtime cannot mutate what they're analyzing.

Alerts are debounced: after the rolling anomaly score crosses the threshold,
a minimum gap of `ALERT_GAP_SEC` must pass before a new alert is emitted. This
keeps the alerts log sparse during a single attack instead of flooding with
duplicates every second.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import joblib
import numpy as np
import torch

from twin_core import (
    INPUT_LEN,
    PV_COLS,
    TARGET_LEN,
    ReplaySource,
    TwinBundle,
    build_plant_window,
)

# ── 4-class attack classifier ────────────────────────────────────────────────

_CLF_LABELS = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}

def _load_attack_classifier():
    """Load the Mixed-experiment XGBoost classifier (best generalisation).

    Searches two locations:
      1. outputs/ relative to the dashboard package (PlantMirror_dashboard/hai-digital-twin/)
      2. outputs/ two levels up — the main repo root (hai-digital-twin/)
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "outputs" / "classifiers",           # dashboard-local copy
        here.parent.parent.parent / "outputs" / "classifiers",  # main repo
    ]
    for base in candidates:
        clf_path = base / "mixed_xgb_classifier.pkl"
        scaler_path = base / "mixed_xgb_scaler.pkl"
        if clf_path.exists() and scaler_path.exists():
            return joblib.load(clf_path), joblib.load(scaler_path)
    return None, None

_attack_clf, _attack_scaler = _load_attack_classifier()


# ── Per-class feature centroids (computed once from train_data.npz) ──────────

# PV column indices inside the 133-column scaled arrays (from metadata.pkl)
_PV_IDX_IN_FULL = [19, 14, 11, 24, 26]  # P1_PIT01, P1_LIT01, P1_FT03Z, P1_TIT01, P1_TIT03

def _compute_class_centroids() -> Optional[Dict[int, np.ndarray]]:
    """Load train_data.npz and return per-class centroid of 30-dim PV feature vectors."""
    here = Path(__file__).resolve().parent
    repo = here.parent.parent.parent
    npz_path = repo / "outputs" / "scaled_split" / "train_data.npz"
    if not npz_path.exists():
        return None
    d = np.load(str(npz_path))
    y = d["y"]                        # (N, 180, 133)
    labels = d["scenario_labels"]     # (N,)
    pv = y[:, :, _PV_IDX_IN_FULL]    # (N, 180, 5)
    centroids: Dict[int, np.ndarray] = {}
    for cls in range(4):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        feats = _extract_features_batch(pv[mask])   # (M, 30)
        centroids[cls] = feats.mean(axis=0)         # (30,)
    return centroids

def _extract_features_batch(traj: np.ndarray) -> np.ndarray:
    """(N, T, 5) → (N, 30) statistical features — batch version."""
    r = traj.transpose(0, 2, 1)   # (N, 5, T)
    parts = [
        r.mean(axis=2), r.std(axis=2), r.min(axis=2), r.max(axis=2),
        np.abs(r).mean(axis=2), np.diff(r, axis=2).mean(axis=2),
    ]
    return np.concatenate(parts, axis=1)   # (N, 30)

_class_centroids: Optional[Dict[int, np.ndarray]] = _compute_class_centroids()


def similarity_to_class(pv_scaled: np.ndarray, class_id: int) -> Optional[float]:
    """Return 0–1 similarity between a (T,5) PV window and the real training centroid.

    1.0 = identical to centroid, lower = further away.
    Returns None if centroids were not loaded.
    """
    if _class_centroids is None or class_id not in _class_centroids:
        return None
    feats = _extract_features(pv_scaled).flatten()   # (30,)
    centroid = _class_centroids[class_id]
    dist = np.linalg.norm(feats - centroid)
    norm = np.linalg.norm(centroid)
    return float(1.0 / (1.0 + dist / max(norm, 1e-8)))


def _extract_features(window: np.ndarray) -> np.ndarray:
    """(T, 5) scaled PV window → 30-dim feature vector (matches sec3_classification.py)."""
    r = window.T  # (5, T)
    feats = np.concatenate([
        r.mean(1), r.std(1), r.min(1), r.max(1),
        np.abs(r).mean(1), np.diff(r, axis=1).mean(1),
    ])
    return feats.reshape(1, -1)


def classify_window(pv_window: np.ndarray) -> str:
    """Predict attack class from a (T, 5) scaled PV window. Returns label string."""
    if _attack_clf is None:
        return "unknown"
    # Use last 180 steps to match training window length
    win = pv_window[-TARGET_LEN:] if len(pv_window) >= TARGET_LEN else pv_window
    feats = _extract_features(win)
    feats_scaled = _attack_scaler.transform(feats)
    pred = int(_attack_clf.predict(feats_scaled)[0])
    return _CLF_LABELS.get(pred, "unknown")


# Rolling buffer length in simulated seconds (what the UI strips show).
BUFFER_SEC = 900                 # 15 min at 1 Hz
# Length of sliding window used to compute the anomaly score that mirrors
# training-time window MSE. 180s matches TARGET_LEN.
SCORE_WINDOW_SEC = TARGET_LEN    # 180
# Minimum gap between alerts during an ongoing anomaly.
ALERT_GAP_SEC = 60


# ── Snapshot passed to "what-if" tabs (Generative / Predictive deep dive) ────

@dataclass(frozen=True)
class TwinSnapshot:
    """Immutable copy of runtime state at one simulated instant."""

    h_plant: torch.Tensor       # (L, 1, hidden) — cloned detached
    pv_twin_scaled: torch.Tensor  # (1, n_pv)
    sim_clock: int
    cursor: int                 # position in the ReplaySource
    scenario: int
    # Keeping a pointer to the source is safe because ReplaySource arrays are
    # read-only for this app.
    src_name: str


# ── Alert record ─────────────────────────────────────────────────────────────

@dataclass
class Alert:
    sim_clock: int            # twin's own clock
    cursor: int               # replay position at firing time
    score: float              # rolling anomaly score that triggered it
    top_pv: str               # PV contributing most to residual at t
    per_pv_mse: Dict[str, float]
    ground_truth: str         # derived from replay's label/target_controller
    predicted_class: str = "unknown"  # 4-class classifier output

    def as_row(self) -> Dict[str, object]:
        return {
            "sim_clock": self.sim_clock,
            "cursor": self.cursor,
            "score": round(self.score, 5),
            "top_pv": self.top_pv,
            "predicted_class": self.predicted_class,
            "ground_truth": self.ground_truth,
        }


# ── Runtime ──────────────────────────────────────────────────────────────────

class TwinRuntime:
    def __init__(self, bundle: TwinBundle, src: ReplaySource):
        self.bundle = bundle
        self.src = src

        # Persistent simulation state
        self.h_plant: Optional[torch.Tensor] = None
        self.pv_twin: Optional[torch.Tensor] = None   # (1, n_pv), scaled
        self.sim_clock: int = 0
        self.cursor: int = INPUT_LEN
        self.scenario: int = 0

        # Rolling buffers (deques so trimming is O(1))
        self.twin_pv: Deque[np.ndarray] = deque(maxlen=BUFFER_SEC)
        self.actual_pv: Deque[np.ndarray] = deque(maxlen=BUFFER_SEC)
        self.per_pv_residual: Deque[np.ndarray] = deque(maxlen=BUFFER_SEC)
        self.sim_times: Deque[int] = deque(maxlen=BUFFER_SEC)
        self.cursor_times: Deque[int] = deque(maxlen=BUFFER_SEC)

        # For the rolling anomaly score (mean of last SCORE_WINDOW_SEC step MSEs)
        self._score_window: Deque[float] = deque(maxlen=SCORE_WINDOW_SEC)

        # Alerts
        self.alerts: List[Alert] = []
        self._last_alert_at: int = -(10 ** 9)

        self.is_ready = False

    # ── Initialization / reset ──────────────────────────────────────────

    def warm_up(self, cursor: Optional[int] = None, scenario: int = 0) -> None:
        """Initialize h_plant + pv_twin from the 300s ending at cursor."""
        if cursor is not None:
            self.cursor = int(cursor)
        self.scenario = int(scenario)

        if self.cursor < INPUT_LEN:
            self.cursor = INPUT_LEN
        if self.cursor >= len(self.src):
            raise ValueError(
                f"Cursor {self.cursor} beyond replay length {len(self.src)}"
            )

        win = build_plant_window(self.bundle, self.src, self.cursor)
        if win is None:
            raise ValueError(
                f"Cannot build warm-up window at cursor={self.cursor}"
            )

        # Use the scenario requested by the caller, not whatever
        # build_plant_window inferred from labels.
        scen_t = torch.tensor(
            [self.scenario], dtype=torch.long, device=self.bundle.device
        )
        self.h_plant = self.bundle.plant.encode_only(win["x_cv"], scen_t)
        self.pv_twin = win["pv_init"].clone()  # (1, n_pv) scaled

        # Clear rolling state
        self.twin_pv.clear()
        self.actual_pv.clear()
        self.per_pv_residual.clear()
        self.sim_times.clear()
        self.cursor_times.clear()
        self._score_window.clear()
        if hasattr(self, "_window_score"):
            self._window_score.clear()
        else:
            self._window_score = deque(maxlen=20)
        self.alerts.clear()
        self._last_alert_at = -(10 ** 9)
        self.sim_clock = 0
        self.is_ready = True

    def reset(self, cursor: Optional[int] = None,
              scenario: Optional[int] = None) -> None:
        """Re-warm at current or given cursor / scenario."""
        self.warm_up(
            cursor=cursor if cursor is not None else self.cursor,
            scenario=scenario if scenario is not None else self.scenario,
        )

    # ── Stepping ────────────────────────────────────────────────────────

    def step(self, n: int = 1) -> int:
        """Advance n simulated seconds. Returns the number of steps actually
        taken (may be < n if we hit the end of the replay)."""
        if not self.is_ready:
            raise RuntimeError("TwinRuntime.step() called before warm_up()")

        bundle = self.bundle
        s = bundle.scalers
        pv_idx = s.pv_idx
        plant_in_idx = s.plant_in_idx
        device = bundle.device

        steps_done = 0
        for _ in range(n):
            t = self.cursor
            if t + 1 > len(self.src):
                break

            # ── 1. Update the authentic Window Score every 5 ticks so the gauge
            #       moves continuously (detector semantics unchanged — still the
            #       same batched 180-s predict MSE, just refreshed more often). ──
            if self.sim_clock % 5 == 0:
                win = build_plant_window(self.bundle, self.src, t)
                if win is not None:
                    pv_pred_t, haiend_pred_t = self.bundle.plant.predict(
                        win["x_cv"], win["x_cv_target"], win["pv_init"], win["scenario"]
                    )
                    pv_pred = pv_pred_t.squeeze(0).cpu().numpy()       # (180, 5)
                    pv_true = win["pv_target"].squeeze(0).cpu().numpy()
                    pv_mse = float(np.mean((pv_pred - pv_true) ** 2))

                    # HAIEND dual-head scoring (matches sec3_detection.py evaluation)
                    haiend_mse = 0.0
                    if (haiend_pred_t is not None
                            and len(self.bundle.haiend_idx) > 0):
                        haiend_pred = haiend_pred_t.squeeze(0).cpu().numpy()   # (180, 36)
                        t0_win = int(win["t1"].item())
                        t2_win = int(win["t2"].item())
                        haiend_true = self.src.scaled[t0_win:t2_win][
                            :, self.bundle.haiend_idx
                        ].astype(np.float32)
                        n = min(haiend_pred.shape[1], haiend_true.shape[1])
                        haiend_mse = float(np.mean(
                            (haiend_pred[:, :n] - haiend_true[:, :n]) ** 2
                        ))

                    # Combined score: 50/50 PV + HAIEND (only when HAIEND available)
                    if haiend_mse > 0.0:
                        win_mse = 0.5 * pv_mse + 0.5 * haiend_mse
                    else:
                        win_mse = pv_mse

                    self._window_score.append(win_mse)
                    per_pv_array = np.mean((pv_pred - pv_true) ** 2, axis=0)
                    self._last_per_pv_mse = {PV_COLS[i]: float(per_pv_array[i]) for i in range(len(PV_COLS))}

            # ── 2. Run the continuous visual stream (which will drift, but we don't alert on it) ──
            x_cv_np = self.src.scaled[t, plant_in_idx]            # (128,)
            x_cv_t = torch.from_numpy(x_cv_np).unsqueeze(0).to(device).float()

            pv_new, _haiend_step, h_new = bundle.plant.step_once(
                x_cv_t, self.h_plant, self.pv_twin
            )
            # Actual (replay) PV at this second
            pv_actual = self.src.scaled[t, pv_idx]                # (5,)

            twin_pv_np = pv_new.squeeze(0).cpu().numpy()          # (5,)
            resid_vec = (twin_pv_np - pv_actual) ** 2             # (5,)
            step_mse = float(resid_vec.mean())

            # Append buffers
            self.twin_pv.append(twin_pv_np)
            self.actual_pv.append(pv_actual)
            self.per_pv_residual.append(resid_vec)
            self.sim_times.append(self.sim_clock)
            self.cursor_times.append(t)
            
            # Anomaly check reading the newly architected _window_score
            self._maybe_fire_alert()

            # Carry state
            self.h_plant = h_new
            self.pv_twin = pv_new
            self.cursor += 1
            self.sim_clock += 1
            steps_done += 1

        return steps_done

    # ── Alerts ──────────────────────────────────────────────────────────

    @property
    def anomaly_score(self) -> Optional[float]:
        """Returns the most recent authentic 180s window score."""
        if not hasattr(self, "_window_score") or not self._window_score:
            return None
        return float(self._window_score[-1])

    def _maybe_fire_alert(self) -> None:
        score = self.anomaly_score
        if score is None:
            return
        if score <= self.bundle.threshold:
            return
        if self.sim_clock - self._last_alert_at < ALERT_GAP_SEC:
            return

        top_pv = "?"
        per_pv_dict = {}
        if hasattr(self, "_last_per_pv_mse"):
            per_pv_dict = self._last_per_pv_mse
            if per_pv_dict:
                top_pv = max(per_pv_dict.items(), key=lambda item: item[1])[0]

        gt = self._ground_truth_at(self.cursor)
        # Run 4-class classifier on the current rolling PV buffer
        pv_class = "unknown"
        if len(self.twin_pv) >= 10:
            pv_arr = np.stack(list(self.twin_pv), axis=0)  # (T, 5) scaled
            pv_class = classify_window(pv_arr)
        self.alerts.append(Alert(
            sim_clock=self.sim_clock,
            cursor=self.cursor,
            score=score,
            top_pv=top_pv,
            per_pv_mse=per_pv_dict,
            ground_truth=gt,
            predicted_class=pv_class,
        ))
        self._last_alert_at = self.sim_clock

    def _ground_truth_at(self, cursor: int) -> str:
        """Label the current cursor using the replay's attack metadata."""
        df = self.src.df_raw
        if "label" not in df.columns:
            return "unknown"
        # Look at the SCORE_WINDOW_SEC seconds ending at cursor
        t0 = max(0, cursor - SCORE_WINDOW_SEC)
        slc = df.iloc[t0:cursor]
        if (slc["label"] == 0).all():
            return "normal"
        attacked = slc[slc["label"] > 0]
        at = ctrl = "?"
        if "attack_type" in attacked.columns and len(attacked):
            at = str(attacked["attack_type"].mode().iloc[0])
        if "target_controller" in attacked.columns and len(attacked):
            vals = attacked["target_controller"].dropna()
            if len(vals):
                ctrl = str(vals.mode().iloc[0])
        return f"ATTACK: {at} @ {ctrl}"

    # ── Snapshots & views ───────────────────────────────────────────────

    def snapshot(self) -> TwinSnapshot:
        """Deep copy of runtime state. Used by Predictive/Generative tabs so
        what-if rollouts are immune to concurrent mutation by step()."""
        if not self.is_ready:
            raise RuntimeError("TwinRuntime.snapshot() called before warm_up()")
        return TwinSnapshot(
            h_plant=self.h_plant.detach().clone(),
            pv_twin_scaled=self.pv_twin.detach().clone(),
            sim_clock=self.sim_clock,
            cursor=self.cursor,
            scenario=self.scenario,
            src_name=self.src.name,
        )

    def rolling_arrays(self) -> Dict[str, np.ndarray]:
        """Return the rolling buffers as numpy arrays for plotting."""
        if not self.twin_pv:
            empty_pv = np.zeros((0, len(PV_COLS)), dtype=np.float32)
            return {
                "twin_pv": empty_pv, "actual_pv": empty_pv,
                "per_pv_residual": empty_pv,
                "step_mse": np.zeros(0, dtype=np.float32),
                "sim_times": np.zeros(0, dtype=np.int64),
                "cursor_times": np.zeros(0, dtype=np.int64),
            }
        twin = np.stack(list(self.twin_pv), axis=0)
        actual = np.stack(list(self.actual_pv), axis=0)
        per_pv = np.stack(list(self.per_pv_residual), axis=0)
        return {
            "twin_pv": twin,
            "actual_pv": actual,
            "per_pv_residual": per_pv,
            "step_mse": per_pv.mean(axis=1),
            "sim_times": np.fromiter(self.sim_times, dtype=np.int64),
            "cursor_times": np.fromiter(self.cursor_times, dtype=np.int64),
        }


# ── Standalone smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    from twin_core import default_paths, load_bundle, load_replay

    paths = default_paths()
    bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
    src = load_replay(paths["test_csvs"][1], bundle.scalers)

    rt = TwinRuntime(bundle, src)
    rt.warm_up(cursor=INPUT_LEN, scenario=0)
    print(f"warm_up OK; sim_clock=0, cursor={rt.cursor}")

    # Advance 600 simulated seconds at 60x speed (10 ticks of 60 sim-sec each)
    for _ in range(10):
        n = rt.step(60)
        r = rt.rolling_arrays()
        score = rt.anomaly_score
        print(f"  +{n:2d} steps  sim_clock={rt.sim_clock:5d}  "
              f"cursor={rt.cursor:6d}  "
              f"score={'n/a' if score is None else f'{score:.5f}'}  "
              f"alerts={len(rt.alerts)}")

    # Advance to a known attack region in test2.csv and look for alerts
    rt.reset(cursor=197000, scenario=0)
    rt.step(600)
    score = rt.anomaly_score
    print(f"\nNear t=197700 (known attack):")
    print(f"  sim_clock={rt.sim_clock}  score={score:.5f}  "
          f"alerts={len(rt.alerts)}")
    for a in rt.alerts[:3]:
        print(f"  alert: {a.as_row()}")

    # Snapshot isolation test
    snap = rt.snapshot()
    pv_before = snap.pv_twin_scaled.clone()
    rt.step(30)
    pv_after = rt.pv_twin.clone()
    diff = (pv_before - snap.pv_twin_scaled).abs().max().item()
    live_moved = (pv_before - pv_after).abs().max().item()
    print(f"\nSnapshot isolation: snap delta = {diff:.3e}  "
          f"(must be 0)  live delta = {live_moved:.3e}  (must be > 0)")
    assert diff == 0.0, "snapshot was mutated by step()"
    print("OK")
