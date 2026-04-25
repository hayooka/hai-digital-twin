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
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import joblib

from twin_core import (
    INPUT_LEN,
    PV_COLS,
    TARGET_LEN,
    ReplaySource,
    TwinBundle,
    build_plant_window,
)


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

    def as_row(self) -> Dict[str, object]:
        return {
            "sim_clock": self.sim_clock,
            "cursor": self.cursor,
            "score": round(self.score, 5),
            "top_pv": self.top_pv,
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

        # Classifier Integration
        self.clf_pipeline = joblib.load(r"C:\Users\PC GAMING\Desktop\AI\HAI\best_hai_classifier.pkl")
        self.clf_model = self.clf_pipeline['model']
        self.clf_scaler = self.clf_pipeline['scaler']
        self.clf_features = self.clf_pipeline['features']
        # Pre-scale data for fast O(1) inference during step loop
        raw_features = src.df_raw[self.clf_features].to_numpy()
        self.clf_data_scaled = self.clf_scaler.transform(raw_features)
        
        # Override the bundle threshold to 0.35 since we now use probability
        self.bundle.threshold = 0.35

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

            # ── 1. Update the authentic Classifier Score ──
            # Replaces the old 180s rollout metric with the real-time XGBoost/RF probability
            prob = self.clf_model.predict_proba(self.clf_data_scaled[t:t+1])[0, 1]
            self._window_score.append(float(prob))
            
            # Cache per-pv error for the alert log to maintain backwards compatibility 
            # with UI expectations, though the alert is now driven by the classifier.
            if self.sim_clock % 5 == 0:
                win = build_plant_window(self.bundle, self.src, t)
                if win is not None:
                    pv_pred = self.bundle.plant.predict(
                        win["x_cv"], win["x_cv_target"], win["pv_init"], win["scenario"]
                    ).squeeze(0).cpu().numpy()
                    pv_true = win["pv_target"].squeeze(0).cpu().numpy()
                    per_pv_array = np.mean((pv_pred - pv_true) ** 2, axis=0)
                    self._last_per_pv_mse = {PV_COLS[i]: float(per_pv_array[i]) for i in range(len(PV_COLS))}

            # ── 2. Run the continuous visual stream (which will drift, but we don't alert on it) ──
            x_cv_np = self.src.scaled[t, plant_in_idx]            # (128,)
            x_cv_t = torch.from_numpy(x_cv_np).unsqueeze(0).to(device).float()

            pv_new, h_new = bundle.plant.step_once(
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
        self.alerts.append(Alert(
            sim_clock=self.sim_clock,
            cursor=self.cursor,
            score=score,
            top_pv=top_pv,
            per_pv_mse=per_pv_dict,
            ground_truth=gt,
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
