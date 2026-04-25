"""
base_twin.py — Minimal core for a new Digital Twin.

Only the mathematical engine (GRU surrogate), the causal‑graph utilities, and the data‑loading pipeline are provided.
No Streamlit UI, no transport bar, no visual panels – this is a pure Python library you can import
or run directly from a script.

Usage examples (at the bottom of the file) show how to:
  • Load the pretrained GRU bundle (weights, scalers, threshold).
  • Load a replay CSV (raw sensor/actuator data).
  • Warm‑up the twin with a 300‑second context.
  • Run a fresh 180‑second rollout using the *exact* same logic the training script used.
  • Compute a per‑PV residual contribution vector.
  • Query the causal graph for upstream suspects.

All paths are relative to the project root – you can adjust `default_paths()` in `twin_core` if you store data elsewhere.
"""

from __future__ import annotations

import numpy as np
import torch
from collections import deque
from pathlib import Path

# Core twin utilities – these are already part of the repo.
from twin_core import (
    build_plant_window,
    default_paths,
    load_bundle,
    load_replay,
    TwinBundle,
    ReplaySource,
)

# Causal graph utilities – also already present.
from causal_utils import CausalGraph

# ---------------------------------------------------------------------------
# Minimal runtime handling ---------------------------------------------------
# ---------------------------------------------------------------------------

class BaseTwin:
    """Lightweight wrapper around the GRU surrogate.

    It mirrors the behaviour of `TwinRuntime` but strips away any UI‑related state.
    The twin maintains an internal GRU hidden state (`h_plant`) and the last
    predicted process‑variable vector (`pv_twin`).
    """

    def __init__(self, bundle: TwinBundle, src: ReplaySource, causal_json: Path | None = None):
        self.bundle = bundle
        self.src = src
        self.cursor = 0               # Index into the replay data
        self.sim_clock = 0            # Simulated time (seconds)
        self.h_plant = None           # GRU hidden state
        self.pv_twin = None           # Last predicted PV vector (torch tensor)
        self._window_scores = deque(maxlen=20)   # Fresh 180‑s rollout scores
        self._last_per_pv = {}
        # Optional causal graph – useful for diagnostics
        if causal_json is not None:
            self.graph = CausalGraph.load(causal_json)
        else:
            self.graph = None

    # ---------------------------------------------------------------------
    # Warm‑up – encode a 300‑second context and initialise the hidden state.
    # ---------------------------------------------------------------------
    def warm_up(self, cursor: int, scenario: int = 0) -> None:
        """Encode the window `[cursor‑INPUT_LEN, cursor)` and set the hidden state.

        Parameters
        ----------
        cursor: int
            Position in the replay at which we want to start the live stream.
        scenario: int, optional
            Scenario embedding index – default ``0`` (normal operation).
        """
        self.cursor = cursor
        self.sim_clock = 0
        # Encode the context – this returns a fresh hidden state for the plant.
        self.h_plant = self.bundle.plant.encode(
            self.src.scaled[cursor - self.bundle.INPUT_LEN : cursor, self.bundle.plant_in_idx]
        )
        # Initialise the twin's PV with the last real measurement in the window.
        self.pv_twin = torch.from_numpy(
            self.src.scaled[cursor - 1, self.bundle.pv_idx]
        ).unsqueeze(0).float().to(self.bundle.device)

    # ---------------------------------------------------------------------
    # Step – advance one simulated second, returning the fresh prediction.
    # ---------------------------------------------------------------------
    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """Run a single autoregressive step.

        Returns
        -------
        twin_pv : np.ndarray (5,)
            The GRU‑predicted process variables for the current second.
        actual_pv : np.ndarray (5,)
            The ground‑truth measurement from the replay.
        """
        t = self.cursor
        # Control vector for this second (scaled)
        x_cv_np = self.src.scaled[t, self.bundle.plant_in_idx]
        x_cv = torch.from_numpy(x_cv_np).unsqueeze(0).float().to(self.bundle.device)
        # One‑step forward using the stored hidden state.
        twin_pv_t, new_h = self.bundle.plant.step_once(x_cv, self.h_plant, self.pv_twin)
        # Ground‑truth PV (scaled)
        actual_pv_np = self.src.scaled[t, self.bundle.pv_idx]
        # Update internal state.
        self.h_plant = new_h
        self.pv_twin = twin_pv_t
        self.cursor += 1
        self.sim_clock += 1
        return twin_pv_t.squeeze(0).cpu().numpy(), actual_pv_np

    # ---------------------------------------------------------------------
    # Fresh 180‑s rollout – the same metric used during training.
    # ---------------------------------------------------------------------
    def fresh_window_score(self, cursor: int, scenario: int = 0) -> float:
        """Compute the 180‑second rollout MSE starting at ``cursor``.

        This is the *exact* statistic the training script used to calibrate the
        detection threshold, so it is the correct value for any alert logic.
        """
        win = build_plant_window(self.bundle, self.src, cursor, scenario=scenario)
        if win is None:
            raise RuntimeError("Unable to build plant window – check cursor bounds.")
        # Run a *full* 180‑s rollout in one shot.
        pred = self.bundle.plant.predict(
            win["x_cv"], win["x_cv_target"], win["pv_init"], win["scenario"]
        ).squeeze(0).cpu().numpy()
        true = win["pv_target"].squeeze(0).cpu().numpy()
        mse = float(np.mean((pred - true) ** 2))
        # Store for later inspection (optional).
        self._window_scores.append(mse)
        # Per‑PV contribution for diagnostics.
        per_pv = np.mean((pred - true) ** 2, axis=0)
        self._last_per_pv = {"PV_" + str(i): float(per_pv[i]) for i in range(len(per_pv))}
        return mse

    # ---------------------------------------------------------------------
    # Diagnostic helpers ----------------------------------------------------
    # ---------------------------------------------------------------------
    def top_residual_pv(self) -> str:
        """Return the PV name with the highest average residual in the last window."""
        if not self._last_per_pv:
            return "N/A"
        return max(self._last_per_pv.items(), key=lambda kv: kv[1])[0]

    def causal_upstream(self, target_pv: str, max_depth: int = 2, level_cap: int = 2):
        """Trace upstream suspects using the loaded causal graph.

        Parameters
        ----------
        target_pv: str
            The PV name you want to investigate (e.g. ``"PV_02"``).
        max_depth: int, optional
            BFS depth limit – default ``2``.
        level_cap: int, optional
            Maximum causal level – default ``2``.
        """
        if self.graph is None:
            raise RuntimeError("Causal graph not loaded – provide a json path to the constructor.")
        return self.graph.rank_suspects(target_pv, max_depth=max_depth, level_cap=level_cap)

# ---------------------------------------------------------------------------
# Example entry‑point (run as a script) ------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Locate default data paths (checkpoint, split, test CSVs).
    paths = default_paths()
    # Load the pretrained GRU bundle.
    bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
    # Choose a test CSV – here we simply pick the first one.
    test_csv = next(p for p in paths["test_csvs"] if p.exists())
    src = load_replay(test_csv, id(bundle))

    # 2️⃣ Initialise the twin.
    twin = BaseTwin(bundle, src, causal_json=paths["causal_json"])
    # Warm‑up at the very first possible cursor (INPUT_LEN).
    twin.warm_up(cursor=bundle.INPUT_LEN, scenario=0)

    # 3️⃣ Run a short live stream (e.g., 100 steps) just to see the predictions.
    print("Running live stream …")
    for _ in range(100):
        pred, real = twin.step()
        resid = np.mean((pred - real) ** 2)
        print(f"t={twin.sim_clock:03d}  MSE={resid:.6f}")

    # 4️⃣ Compute a fresh 180‑s window score starting at the current cursor.
    window_score = twin.fresh_window_score(twin.cursor, scenario=0)
    print(f"\nFresh 180‑s rollout MSE (the statistic the detector expects): {window_score:.6f}")
    print("Top residual PV:", twin.top_residual_pv())

    # 5️⃣ Optional causal trace for the top‑PV.
    if twin.graph is not None:
        suspects = twin.causal_upstream(twin.top_residual_pv())
        print("\nUpstream causal suspects (ordered by score):")
        for s, score, path in suspects[:5]:
            print(f"  {s:<12} score={score:.4f}   path={' → '.join(e.parent for e in path)}")

    print("\nDone.  You now have a pure GRU + causal core you can import in any other project.")
