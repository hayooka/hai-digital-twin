"""
data_loader.py — cached loaders for the Final Dashboard.

All heavy artifacts (plant bundle, XGBoost classifier, causal graph, replay CSV,
pre-cached predict_proba) are loaded ONCE via @st.cache_resource / @st.cache_data
so the dashboard doesn't reload on every interaction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make sibling packages importable (generator/, attack_sim/).
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "generator"))
sys.path.insert(0, str(REPO / "attack_sim"))

import torch  # noqa: E402

# `core` and `attacks` resolve after sys.path is patched.
from core import load_bundle, load_replay, default_paths, LOOP_SPECS, PV_COLS, INPUT_LEN, TARGET_LEN  # noqa: E402
from attacks import AttackSpec, InjectionPoint, AttackType, run_attack_sim  # noqa: E402


# ── Paths (canonical locations, hardcoded — not parametric) ──────────────

CAUSAL_GRAPH_PATH = REPO / "outputs" / "causal_graph" / "parents_full.json"
BENCH_JSON_PATH   = REPO / "outputs" / "classifier" / "bench.json"
XGBOOST_PATH      = Path(r"C:\Users\PC GAMING\Desktop\AI\HAI\best_hai_classifier.pkl")
TEST1_CSV         = Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test1.csv")
TEST2_CSV         = Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test2.csv")
EVAL_RESULTS_PATH = REPO / "generator" / "weights" / "eval_results.json"
TWIN_RESULTS_PATH = REPO / "generator" / "weights" / "results.json"


# ── Plant bundle (5 controllers + plant + scalers) ────────────────────────

@st.cache_resource(show_spinner="Loading GRU plant + 5 controllers…")
def get_bundle():
    """Loaded once per session."""
    paths = default_paths()
    bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
    return bundle


@st.cache_resource(show_spinner="Loading test replay (test1.csv)…")
def get_replay():
    bundle = get_bundle()
    return load_replay(TEST1_CSV, bundle.scalers)


# ── XGBoost Guardian ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading XGBoost Guardian…")
def get_guardian() -> Dict[str, Any]:
    if not XGBOOST_PATH.exists():
        return {"available": False, "path": str(XGBOOST_PATH)}
    pipe = joblib.load(XGBOOST_PATH)
    return {"available": True, **pipe}  # scaler, model, features


@st.cache_data(show_spinner="Scoring real test set with Guardian (~10s)…")
def get_guardian_scores():
    """
    Run Guardian once on the full real test set (test1 + test2) and cache
    proba + y_true. The threshold slider in Tab 4 can then recompute
    F1/Precision/Recall live with ~zero latency.
    """
    g = get_guardian()
    if not g["available"]:
        return None
    df = pd.concat(
        [pd.read_csv(TEST1_CSV, low_memory=False),
         pd.read_csv(TEST2_CSV, low_memory=False)],
        ignore_index=True,
    )
    features = g["features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(f"XGBoost expects missing features: {missing[:5]}")
    X = df[features].to_numpy(dtype=np.float32)
    X_s = g["scaler"].transform(X)
    proba = g["model"].predict_proba(X_s)[:, 1].astype(np.float32)
    y_true = (df["label"] > 0).astype(np.int8).to_numpy()
    return {"proba": proba, "y_true": y_true, "n_rows": len(y_true),
            "n_attack": int(y_true.sum()), "n_normal": int((y_true == 0).sum())}


# ── Causal graph ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_causal_graph() -> Dict[str, List[Dict]]:
    if not CAUSAL_GRAPH_PATH.exists():
        return {}
    with open(CAUSAL_GRAPH_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_coupling_matrix() -> Dict[str, Any]:
    """
    Derive 5x5 loop-to-loop coupling from parents_full.json.
    Sum of (1 / (1 + lag + level)) over all edges that cross loop boundaries.

    Aliases close-cousin column names so FC (P1_FT03Z = same physical signal as
    P1_FT03 in the graph) and CC (P1_PP04 = same actuator as P1_PP04D) actually
    populate.
    """
    graph = get_causal_graph()
    LOOPS = list(LOOP_SPECS.keys())
    col_to_loop: Dict[str, str] = {}
    for loop, spec in LOOP_SPECS.items():
        for c in (spec["base_cols"] + spec["causal_cols"]):  # type: ignore[operator]
            col_to_loop.setdefault(c, loop)

    # ── Aliases ────────────────────────────────────────────────────────
    # The causal graph uses raw / pre-scaling sensor names; LOOP_SPECS uses
    # the controller-facing zero-corrected / digital-command variants.
    # Same physical channels, different column suffix.
    ALIASES = {
        "P1_FT03":   "FC",   # raw flow reading -> FC's PV is P1_FT03Z
        "P1_PP04D":  "CC",   # pump digital command -> CC's CV is P1_PP04
        "P1_PP04":   "CC",
        "P1_FT03Z":  "FC",
    }
    for col, loop in ALIASES.items():
        col_to_loop.setdefault(col, loop)

    mat = np.zeros((len(LOOPS), len(LOOPS)), dtype=np.float32)
    for child, plist in graph.items():
        c_loop = col_to_loop.get(child)
        if c_loop is None:
            continue
        for p in plist:
            p_loop = col_to_loop.get(p["parent"])
            if p_loop is None or p_loop == c_loop:
                continue
            weight = 1.0 / (1.0 + int(p.get("lag", 0)) + int(p.get("level", 0)))
            mat[LOOPS.index(p_loop), LOOPS.index(c_loop)] += weight
    return {"matrix": mat.tolist(), "loops": LOOPS}


@st.cache_data(show_spinner=False)
def get_response_delays() -> Dict[str, int]:
    """CV -> PV max lag per loop, from parents_full.json (with aliases)."""
    graph = get_causal_graph()
    # If the canonical PV name isn't in the graph, fall back to its raw cousin.
    PV_ALIAS = {"P1_FT03Z": "P1_FT03"}   # FC's controller PV vs graph's raw PV
    out: Dict[str, int] = {}
    for loop, spec in LOOP_SPECS.items():
        pv = spec["base_cols"][1]   # type: ignore[index]
        parents = graph.get(pv) or graph.get(PV_ALIAS.get(pv, ""), [])
        if parents:
            out[loop] = max(int(p.get("lag", 0)) for p in parents)
        else:
            out[loop] = 0
    return out


# ── Metrics + eval JSONs ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_eval_results() -> Dict[str, Any]:
    if not EVAL_RESULTS_PATH.exists():
        return {}
    with open(EVAL_RESULTS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_twin_results() -> Dict[str, Any]:
    if not TWIN_RESULTS_PATH.exists():
        return {}
    with open(TWIN_RESULTS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_bench_json() -> Dict[str, Any]:
    if not BENCH_JSON_PATH.exists():
        return {}
    with open(BENCH_JSON_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_eval_1800s() -> Optional[Dict[str, Any]]:
    """30-min horizon NRMSE produced by `run_long_horizon_eval.py`."""
    p = HERE / "cache" / "eval_1800s.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ── XGBoost Hybrid Guardian — production detector used by Tab 1 saved-attacks panel ──

@st.cache_resource(show_spinner="Loading XGBoost Hybrid Guardian…")
def get_trts_rf() -> Dict[str, Any]:
    """Load the production Guardian and return {scaler, model, features, threshold}.

    Despite the legacy function name, this returns the XGBoost Hybrid Guardian
    (best_hai_classifier.pkl), the project's primary detector — F1 0.587, AUROC
    0.904 at peak-F1 threshold 0.35. Trained on Mixed (real_train + synthetic).
    """
    g = get_guardian()
    if not g.get("available"):
        raise RuntimeError(f"Guardian pickle missing at {XGBOOST_PATH}")
    return {
        "scaler":    g["scaler"],
        "model":     g["model"],
        "features":  list(g["features"]),
        "threshold": 0.35,
        "name":      "XGBoost Hybrid Guardian",
        "n_train":   235_078,
        "n_train_attack": 33_892,
    }


# ── Attack gallery (optional — pre-rendered NPZ, if precompute ran) ──────

@st.cache_data(show_spinner=False)
def get_attack_gallery() -> Optional[Dict[str, Any]]:
    npz_path = HERE / "cache" / "attack_gallery.npz"
    idx_path = HERE / "cache" / "attack_gallery_index.json"
    if not (npz_path.exists() and idx_path.exists()):
        return None
    gal = np.load(npz_path)
    with open(idx_path) as f:
        index = json.load(f)
    return {
        "baseline_pv": gal["baseline_pv"],
        "attacked_pv": gal["attacked_pv"],
        "attack_label": gal["attack_label"],
        "index": index["items"],
        "meta": index.get("meta", {}),
    }


# ── Re-exports for convenience ────────────────────────────────────────────

__all__ = [
    "get_bundle", "get_replay", "get_guardian", "get_guardian_scores",
    "get_causal_graph", "get_coupling_matrix", "get_response_delays",
    "get_eval_results", "get_eval_1800s", "get_twin_results", "get_bench_json",
    "get_attack_gallery", "get_trts_rf",
    "AttackSpec", "InjectionPoint", "AttackType", "run_attack_sim",
    "LOOP_SPECS", "PV_COLS", "INPUT_LEN", "TARGET_LEN",
]
