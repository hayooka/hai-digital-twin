"""
attacks.py — attack injection engine for the frozen HAI digital twin.

Implements three ICS-attack injection points on top of the generator's
closed-loop inference stack:

    SP  — setpoint spoofing (modifies the SP channel the controller sees)
    CV  — actuator command override (modifies the controller output before plant)
    PV  — sensor-feedback spoofing (modifies the PV the controller sees)

Semantics
─────────
Every attack has a single continuous window in absolute (relative-to-cursor)
time:

      [ start_offset ,  start_offset + duration )

    start_offset = 0   → attack begins exactly at the cursor (t_end)
    start_offset < 0   → attacker was already active for |start_offset| seconds
                         before the cursor (attack bleeds into the 300-s
                         history window the controllers see)
    start_offset > 0   → attack begins that many seconds into the 180-s
                         target window (controllers finished planning already;
                         only the plant sees the injection for CV/SP)

The attack is clipped to whichever region each injection point can actually
reach:
    CV → target window only  [0, TARGET_LEN)
    SP → history + target    [-INPUT_LEN, TARGET_LEN)
    PV → history + target    [-INPUT_LEN, TARGET_LEN)

V1 closed-loop limitation
─────────────────────────
Controllers run ONCE on the 300-s history window and emit a fixed 180-s CV
schedule.  So a PV attack that lives entirely inside the target window only
affects the operator view — the controller can't react per-second.  To get a
true closed-loop divergence on a PV attack, use start_offset < 0 so the
spoofed PV enters the controller's history window.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GEN_DIR = HERE.parent / "generator"
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

from core import (  # type: ignore  # noqa: E402
    INPUT_LEN,
    LOOP_ORDER,
    LOOP_SPECS,
    PV_COLS,
    TARGET_LEN,
    ReplaySource,
    TwinBundle,
    closed_loop_rollout,
)


# ── Spec enums ──────────────────────────────────────────────────────────────

class InjectionPoint(str, Enum):
    SP = "SP"
    CV = "CV"
    PV = "PV"


class AttackType(str, Enum):
    BIAS = "bias"      # add `magnitude` (physical units) to the channel
    FREEZE = "freeze"  # hold channel at the value just before the attack
    REPLAY = "replay"  # copy values from `magnitude` seconds earlier


@dataclass
class AttackSpec:
    target_loop: str
    injection_point: InjectionPoint
    attack_type: AttackType
    start_offset: int = 0     # seconds relative to t_end (see module docstring)
    duration: int = 60        # seconds, must be > 0
    magnitude: float = 0.0    # physical-units offset (BIAS) or lag seconds (REPLAY)

    # ── Window helpers ──────────────────────────────────────────────────
    # All windows are returned as half-open [s, e) with e > s; an empty window
    # is (0, 0).

    def target_window(self) -> Tuple[int, int]:
        """Part of the attack landing inside the 180-s target [0, TARGET_LEN)."""
        s = max(0, int(self.start_offset))
        e = min(TARGET_LEN, int(self.start_offset) + max(1, int(self.duration)))
        return (s, e) if e > s else (0, 0)

    def history_window(self) -> Tuple[int, int]:
        """Part of the attack landing inside the 300-s history [0, INPUT_LEN)."""
        # history-index i corresponds to relative time i - INPUT_LEN
        s = max(0, int(self.start_offset) + INPUT_LEN)
        e = min(INPUT_LEN, int(self.start_offset) + max(1, int(self.duration)) + INPUT_LEN)
        return (s, e) if e > s else (0, 0)

    def combined_window(self) -> Tuple[int, int]:
        """Attack window in the joined (history + target) 480-element array."""
        s = max(0, int(self.start_offset) + INPUT_LEN)
        e = min(INPUT_LEN + TARGET_LEN,
                int(self.start_offset) + max(1, int(self.duration)) + INPUT_LEN)
        return (s, e) if e > s else (0, 0)


# ── Transform ───────────────────────────────────────────────────────────────

def _inject_transform(arr: np.ndarray, window: Tuple[int, int],
                      spec: AttackSpec) -> np.ndarray:
    """
    Apply the attack's transform to `arr[s:e]` and return a copy.  `arr` should
    be a 1-D array in physical units (or consistent units with `magnitude`).
    Untouched outside the window.
    """
    out = arr.copy()
    s, e = window
    if s >= e:
        return out

    if spec.attack_type == AttackType.BIAS:
        out[s:e] = out[s:e] + float(spec.magnitude)
        return out

    if spec.attack_type == AttackType.FREEZE:
        # Hold at the value just before the attack; if the attack starts at
        # index 0, freeze at the first in-window value (self-freeze).
        freeze_val = float(out[s - 1]) if s > 0 else float(out[s])
        out[s:e] = freeze_val
        return out

    if spec.attack_type == AttackType.REPLAY:
        lag = max(1, int(spec.magnitude))
        # Copy arr[s-lag : e-lag] into arr[s:e].  If that source window extends
        # before arr[0], pad the un-reachable portion with arr[0] so the
        # attack still behaves like "stale data from the past".
        src_s = s - lag
        src_e = e - lag
        if src_e <= 0:
            # Whole replay source is out-of-bounds → fallback to freeze.
            out[s:e] = float(out[0])
            return out
        if src_s < 0:
            pad_len = -src_s
            # Segment we CAN pull from arr starts at 0
            # and is of length (src_e - 0) = src_e.
            # We map that into the end of the window; the unreachable prefix
            # of the window gets filled with arr[0].
            out[s:s + pad_len] = float(out[0])
            out[s + pad_len:e] = out[0:src_e]
        else:
            out[s:e] = out[src_s:src_e]
        return out

    raise ValueError(f"Unknown attack_type: {spec.attack_type}")


# ── The main engine ─────────────────────────────────────────────────────────

@dataclass
class AttackResult:
    baseline: Dict[str, np.ndarray]
    attacked: Dict[str, np.ndarray]
    spec: AttackSpec
    t_end: int
    scenario: int
    signals: Dict[str, np.ndarray] = field(default_factory=dict)
    attack_label: np.ndarray = field(default_factory=lambda: np.zeros(TARGET_LEN, dtype=np.int8))


def _build_attack_label(spec: AttackSpec) -> np.ndarray:
    """Per-second label over the 180-s target window (1 during attack, else 0)."""
    lbl = np.zeros(TARGET_LEN, dtype=np.int8)
    s, e = spec.target_window()
    lbl[s:e] = 1
    return lbl


def run_attack_sim(
    bundle: TwinBundle,
    src: ReplaySource,
    t_end: int,
    spec: AttackSpec,
    scenario: int = 0,
) -> Optional[AttackResult]:
    """
    Run both the clean baseline and the attacked rollout.  Returns None if the
    cursor is out of range for a full 300 + 180-s window.
    """
    if spec.target_loop not in LOOP_ORDER:
        raise ValueError(f"Unknown target_loop: {spec.target_loop}")
    if spec.duration < 1:
        raise ValueError("duration must be >= 1 second")

    t0 = t_end - INPUT_LEN
    t2 = t_end + TARGET_LEN
    if t0 < 0 or t2 > len(src):
        return None

    # 1. Baseline — unmodified closed-loop rollout
    baseline = closed_loop_rollout(bundle, src, t_end, scenario=scenario)
    if baseline is None:
        return None

    # 2. Attacked — inject at the right layer and re-run what's needed
    attacked = _rollout_with_injection(bundle, src, t_end, spec, scenario)
    if attacked is None:
        return None

    # 3. Extract paired signals for the target loop
    signals = _extract_signals(bundle, src, t_end, spec, baseline, attacked)

    return AttackResult(
        baseline=baseline,
        attacked=attacked,
        spec=spec,
        t_end=t_end,
        scenario=scenario,
        signals=signals,
        attack_label=_build_attack_label(spec),
    )


# ── Rollout with injection ──────────────────────────────────────────────────

def _rollout_with_injection(
    bundle: TwinBundle,
    src: ReplaySource,
    t_end: int,
    spec: AttackSpec,
    scenario: int,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Specialised rollout:
      • SP / PV injection → rewrite the relevant sensor column in the 480-s
        joined (history + target) slice BEFORE running controllers + plant.
      • CV injection → run controllers + plant first, then override the target
        loop's CV in the target window and re-run the plant only.
    """
    t0 = t_end - INPUT_LEN
    t2 = t_end + TARGET_LEN

    loop = spec.target_loop
    cfg = LOOP_SPECS[loop]
    sp_col = cfg["base_cols"][0]   # type: ignore[index]
    pv_col = cfg["base_cols"][1]   # type: ignore[index]
    cv_col = cfg["cv_col"]         # type: ignore[index]

    raw = src.df_raw.iloc[t0:t2].copy()
    combined_s, combined_e = spec.combined_window()

    # ── SP / PV injection into the joined timeline ─────────────────────
    if spec.injection_point == InjectionPoint.SP and combined_e > combined_s:
        combined = raw[sp_col].to_numpy(dtype=np.float64)
        spoofed = _inject_transform(combined, (combined_s, combined_e), spec)
        raw[sp_col] = spoofed

    elif spec.injection_point == InjectionPoint.PV and combined_e > combined_s:
        # Only the HISTORY portion matters for controller behaviour — in V1 the
        # controller doesn't re-plan during the target.  We still rewrite the
        # history portion of the PV column so the controller sees the spoof.
        # The target-window operator view is built separately in _extract_signals.
        hist_s = max(0, combined_s)
        hist_e = min(INPUT_LEN, combined_e)
        if hist_e > hist_s:
            combined = raw[pv_col].to_numpy(dtype=np.float64)
            # Inject in the history portion only; leave target PV real.
            spoofed = _inject_transform(combined, (hist_s, hist_e), spec)
            raw[pv_col] = spoofed

    # ── Run full closed-loop on the (possibly mutated) raw frame ───────
    attacked = _closed_loop_from_raw(bundle, raw, scenario)
    if attacked is None:
        return None

    # ── CV injection — override and re-run plant ───────────────────────
    if spec.injection_point == InjectionPoint.CV:
        s, e = spec.target_window()
        if e > s:
            scalers = bundle.scalers
            cv_scaled_clean = attacked["ctrl_cv_scaled"][loop].copy()
            cv_physical_clean = scalers.inverse_ctrl_cv(loop, cv_scaled_clean)
            cv_physical_attacked = _inject_transform(cv_physical_clean, (s, e), spec)

            # Back into ctrl-scaled space for the output dict
            cs = scalers.ctrl[loop]
            mean_cv = float(cs["mean"][2])    # type: ignore[index]
            scale_cv = float(cs["scale"][2])  # type: ignore[index]
            cv_scaled_attacked = (cv_physical_attacked - mean_cv) / scale_cv

            # Splice attacked CV into plant's x_cv_target in plant-scaled space
            x_cv_tgt = attacked["x_cv_target_used"].copy()
            cv_sensor_idx = scalers.sensor_cols.index(cv_col)
            cv_plant_idx = int(np.where(scalers.plant_in_idx == cv_sensor_idx)[0][0])
            plant_mean = float(scalers.plant_mean[cv_sensor_idx])
            plant_scale = float(scalers.plant_scale[cv_sensor_idx])
            x_cv_tgt[:, cv_plant_idx] = (cv_physical_attacked - plant_mean) / plant_scale

            # Re-run the plant with the attacked x_cv_target
            scaled_local = scalers.scale_plant_row(raw)
            x_cv_in = scaled_local[:INPUT_LEN, scalers.plant_in_idx]
            pv_init = scaled_local[INPUT_LEN - 1, scalers.pv_idx]
            device = bundle.device
            with torch.no_grad():
                pv_scaled = bundle.plant.predict(
                    torch.from_numpy(x_cv_in).unsqueeze(0).to(device).float(),
                    torch.from_numpy(x_cv_tgt).unsqueeze(0).to(device).float(),
                    torch.from_numpy(pv_init).unsqueeze(0).to(device).float(),
                    torch.tensor([scenario], dtype=torch.long, device=device),
                ).squeeze(0).cpu().numpy()
            attacked["pv_scaled"] = pv_scaled
            attacked["pv_physical"] = scalers.inverse_plant(pv_scaled, PV_COLS)
            attacked["ctrl_cv_scaled"][loop] = cv_scaled_attacked
            attacked["x_cv_target_used"] = x_cv_tgt

    return attacked


def _closed_loop_from_raw(
    bundle: TwinBundle,
    raw,
    scenario: int,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Internal close of core.closed_loop_rollout that accepts a pre-built raw
    slice instead of (src, t_end).  Lets us inject SP/PV perturbations into
    the joined history+target before inference.
    """
    scalers = bundle.scalers
    scaled_local = scalers.scale_plant_row(raw)
    x_cv_in = scaled_local[:INPUT_LEN, scalers.plant_in_idx]
    x_cv_tgt = scaled_local[INPUT_LEN:, scalers.plant_in_idx].copy()
    pv_init = scaled_local[INPUT_LEN - 1, scalers.pv_idx]

    ctrl_cv_preds: Dict[str, np.ndarray] = {}
    for loop in LOOP_ORDER:
        cfg = LOOP_SPECS[loop]
        base_df = raw.iloc[:INPUT_LEN]
        base_scaled = scalers.scale_ctrl(loop, base_df)
        causal = scaled_local[
            :INPUT_LEN,
            np.array([scalers.sensor_cols.index(c)
                      for c in cfg["causal_cols"]],  # type: ignore[arg-type]
                     dtype=np.int64),
        ]
        ctrl_in = np.concatenate([base_scaled, causal], axis=1).astype(np.float32)
        ctrl_in_t = torch.from_numpy(ctrl_in).unsqueeze(0).to(bundle.device)
        with torch.no_grad():
            cv_scaled = bundle.controllers[loop].predict(
                ctrl_in_t, target_len=TARGET_LEN
            ).squeeze(0).squeeze(-1).cpu().numpy()
        ctrl_cv_preds[loop] = cv_scaled

        cv_col = cfg["cv_col"]                          # type: ignore[assignment]
        cv_sensor_idx = scalers.sensor_cols.index(cv_col)
        cv_plant_idx = int(np.where(scalers.plant_in_idx == cv_sensor_idx)[0][0])
        cv_physical = scalers.inverse_ctrl_cv(loop, cv_scaled)
        plant_mean = scalers.plant_mean[cv_sensor_idx]
        plant_scale = scalers.plant_scale[cv_sensor_idx]
        x_cv_tgt[:, cv_plant_idx] = (cv_physical - plant_mean) / plant_scale

    device = bundle.device
    with torch.no_grad():
        pv_scaled = bundle.plant.predict(
            torch.from_numpy(x_cv_in).unsqueeze(0).to(device).float(),
            torch.from_numpy(x_cv_tgt).unsqueeze(0).to(device).float(),
            torch.from_numpy(pv_init).unsqueeze(0).to(device).float(),
            torch.tensor([scenario], dtype=torch.long, device=device),
        ).squeeze(0).cpu().numpy()
    pv_physical = scalers.inverse_plant(pv_scaled, PV_COLS)

    return {
        "pv_scaled": pv_scaled,
        "pv_physical": pv_physical,
        "ctrl_cv_scaled": ctrl_cv_preds,
        "x_cv_target_used": x_cv_tgt,
        "scenario": scenario,
    }


# ── Paired-signal extraction ────────────────────────────────────────────────

def _extract_signals(
    bundle: TwinBundle,
    src: ReplaySource,
    t_end: int,
    spec: AttackSpec,
    baseline: Dict[str, np.ndarray],
    attacked: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Pack the target loop's SP/CV/PV pairs (real vs seen) over the 180-s target
    window for plotting + CSV export.  Units are physical for all three pairs.
    """
    loop = spec.target_loop
    cfg = LOOP_SPECS[loop]
    sp_col = cfg["base_cols"][0]   # type: ignore[index]
    pv_col = cfg["base_cols"][1]   # type: ignore[index]

    t1 = t_end
    t2 = t_end + TARGET_LEN

    # SP pair
    sp_real = src.df_raw[sp_col].to_numpy(dtype=np.float64)[t1:t2].copy()
    sp_seen = sp_real.copy()
    if spec.injection_point == InjectionPoint.SP:
        # Apply the transform over the full joined timeline, then take the
        # target portion — matches what the plant/controller actually saw.
        combined_real = src.df_raw[sp_col].to_numpy(dtype=np.float64)[t1 - INPUT_LEN:t2]
        combined_seen = _inject_transform(combined_real, spec.combined_window(), spec)
        sp_seen = combined_seen[INPUT_LEN:]

    # CV pair
    cv_intended = bundle.scalers.inverse_ctrl_cv(loop, baseline["ctrl_cv_scaled"][loop])
    cv_actual = bundle.scalers.inverse_ctrl_cv(loop, attacked["ctrl_cv_scaled"][loop])

    # PV pair
    pv_idx = PV_COLS.index(pv_col) if pv_col in PV_COLS else None
    if pv_idx is None:
        pv_real = np.full(TARGET_LEN, np.nan)
        pv_seen = np.full(TARGET_LEN, np.nan)
    else:
        pv_real = attacked["pv_physical"][:, pv_idx]
        pv_seen = pv_real.copy()
        if spec.injection_point == InjectionPoint.PV:
            # Operator-view spoof in the target window: apply the attack
            # transform to the attacked-plant PV over the joined timeline
            # (history portion uses src to fill the reference).
            hist_ref = src.df_raw[pv_col].to_numpy(dtype=np.float64)[t1 - INPUT_LEN:t1]
            combined_seen = np.concatenate([hist_ref, pv_real])
            combined_seen = _inject_transform(combined_seen, spec.combined_window(), spec)
            pv_seen = combined_seen[INPUT_LEN:]

    return {
        "SP_real": sp_real, "SP_seen": sp_seen,
        "CV_intended": cv_intended, "CV_actual": cv_actual,
        "PV_real": pv_real, "PV_seen": pv_seen,
    }


# ── CLI smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from core import default_paths, load_bundle, load_replay  # type: ignore

    paths = default_paths()
    bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
    src = load_replay(paths["test_csvs"][0], bundle.scalers)

    t_end = INPUT_LEN + 1000
    configs = [
        # (injection, attack_type, start_offset, duration, magnitude)
        ("SP", "bias",   0,   60, 20.0),
        ("SP", "freeze", -30, 60, 0.0),
        ("SP", "replay", 0,   60, 30.0),
        ("CV", "bias",   30,  60, 10.0),
        ("CV", "freeze", 0,   90, 0.0),
        ("CV", "replay", 0,   60, 30.0),
        ("PV", "bias",   -60, 90, 20.0),
        ("PV", "freeze", -60, 60, 0.0),
        ("PV", "replay", -60, 60, 30.0),
    ]
    print(f"{'inj':4s} {'type':6s} {'start':>5s} {'dur':>4s} {'mag':>6s} | "
          f"{'dSP':>7s} {'dCV':>7s} {'dPV':>7s}")
    print("-" * 70)
    for ip, at, so, du, mg in configs:
        spec = AttackSpec(
            target_loop="LC",
            injection_point=InjectionPoint(ip),
            attack_type=AttackType(at),
            start_offset=so, duration=du, magnitude=mg,
        )
        r = run_attack_sim(bundle, src, t_end=t_end, spec=spec)
        assert r is not None, f"{ip}/{at} cursor out of range"
        sig = r.signals
        d_sp = float(np.max(np.abs(sig["SP_seen"] - sig["SP_real"])))
        d_cv = float(np.max(np.abs(sig["CV_actual"] - sig["CV_intended"])))
        d_pv = float(np.max(np.abs(r.attacked["pv_physical"] - r.baseline["pv_physical"])))
        print(f"{ip:4s} {at:6s} {so:>5d} {du:>4d} {mg:>6.1f} | "
              f"{d_sp:>7.3f} {d_cv:>7.3f} {d_pv:>7.3f}")
    print("OK")
