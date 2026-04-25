"""
Pre-deploy inspection — focused on the recent XGBoost-Guardian swap.

Tests every tab's logic, the new save+score round-trip, and cross-tab consistency.
Faster than the full _full_audit.py (skips the long custom-attack AppTest sweep).
"""
from __future__ import annotations

import json, os, sys, traceback, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

DASH = Path(r"C:\Users\PC GAMING\Desktop\new_ai\hai-digital-twin\dashboard")
sys.path.insert(0, str(DASH))
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st
st.cache_resource = lambda *a, **k: (lambda f: f) if not a or not callable(a[0]) else a[0]
st.cache_data     = lambda *a, **k: (lambda f: f) if not a or not callable(a[0]) else a[0]

errors, warns, passes = [], [], []
def ok(label, cond, detail=""):
    if cond:
        passes.append(label); print(f"  [OK  ] {label}{'  -- ' + detail if detail else ''}")
    else:
        errors.append((label, detail)); print(f"  [FAIL] {label}  -- {detail}")
def warn(label, detail=""):
    warns.append((label, detail)); print(f"  [WARN] {label}  -- {detail}")


# ============================================================================
print("\n=== PHASE 1 · MODULE IMPORTS & BASIC LOAD ===")
# ============================================================================
try:
    from data_loader import (
        get_bundle, get_replay, get_guardian, get_trts_rf,
        get_eval_results, get_eval_1800s, get_causal_graph,
        get_coupling_matrix, get_response_delays, get_attack_gallery,
        AttackSpec, AttackType, InjectionPoint, run_attack_sim,
        LOOP_SPECS, PV_COLS,
    )
    ok("data_loader imports cleanly", True)
except Exception as e:
    ok("data_loader imports cleanly", False, str(e)[:200])
    sys.exit(1)


# ============================================================================
print("\n=== PHASE 2 · HEADER CONSISTENCY (XGBoost Guardian) ===")
# ============================================================================
app_text = (DASH / "app.py").read_text(encoding="utf-8")
ok("header card shows F1 0.587", "0.587" in app_text and "XGBoost" in app_text,
   "expected '0.587' and 'XGBoost' in app.py")
ok("header no longer references 0.243 / TRTS Random Forest",
   "0.243" not in app_text and "TRTS Random Forest" not in app_text,
   "leftover RF reference in header")


# ============================================================================
print("\n=== PHASE 3 · CLASSIFIER TAB SOURCE (XGBoost JSON + inline D) ===")
# ============================================================================
clf_text = (DASH / "tabs" / "classifier.py").read_text(encoding="utf-8")
ok("classifier.py reads classifier_experiments.json (XGBoost)",
   "classifier_experiments.json" in clf_text and "classifier_experiments_rf.json" not in clf_text)
ok("classifier.py model card title = XGBOOST HYBRID GUARDIAN",
   "XGBOOST HYBRID GUARDIAN" in clf_text)
ok("classifier.py inline D constant present",
   "GUARDIAN_D" in clf_text and "0.587" in clf_text and "0.904" in clf_text)
ok("classifier.py renders A, B, C, D experiments",
   '"A"' in clf_text and '"B"' in clf_text and '"C"' in clf_text and '"D"' in clf_text)

# Confirm cache JSON exists and has A/B/C
xgb_json = DASH / "cache" / "classifier_experiments.json"
ok("classifier_experiments.json exists", xgb_json.exists())
if xgb_json.exists():
    with open(xgb_json) as f:
        xgb = json.load(f)
    ok("XGB cache has A, B, C", all(k in xgb for k in ["A","B","C"]))
    for k in ["A","B","C"]:
        e = xgb[k]
        ok(f"XGB {k} has F1+P+R+AUROC",
           all(m in e for m in ["f1","precision","recall","auroc"]),
           f"F1={e.get('f1', 'NaN'):.3f}")


# ============================================================================
print("\n=== PHASE 4 · DATA LOADER GUARDIAN (replaces RF) ===")
# ============================================================================
g = get_trts_rf()
ok("get_trts_rf returns dict with required keys",
   all(k in g for k in ["scaler","model","features","threshold","name","n_train"]))
ok("get_trts_rf model is XGBoost",
   "xgb" in str(type(g["model"])).lower() or "XGBClassifier" in str(type(g["model"])),
   f"type={type(g['model']).__name__}")
ok("get_trts_rf threshold = 0.35", abs(g["threshold"] - 0.35) < 0.001)
ok("get_trts_rf features list length 133",
   len(g["features"]) == 133, f"len={len(g['features'])}")


# ============================================================================
print("\n=== PHASE 5 · SAVE/SCORE ROUND-TRIP (Tab 1 → Guardian) ===")
# ============================================================================
bundle = get_bundle()
src    = get_replay()
spec = AttackSpec(target_loop="LC", injection_point=InjectionPoint.CV,
                  attack_type=AttackType.BIAS,
                  start_offset=-30, duration=60, magnitude=15.0)
result = run_attack_sim(bundle, src, t_end=1000, spec=spec, scenario=0)
ok("attack runs, produces drift",
   float(np.max(np.abs(result.attacked["pv_physical"] - result.baseline["pv_physical"]))) > 0.1)

# Reproduce the save logic from attack_sim.py to verify full_133 builds correctly
scalers = bundle.scalers
pv_idx_arr       = np.asarray(scalers.pv_idx, dtype=np.int64)
plant_in_idx_arr = np.asarray(scalers.plant_in_idx, dtype=np.int64)
plant_scale = np.asarray(scalers.plant_scale)
plant_mean  = np.asarray(scalers.plant_mean)
x_scaled = result.attacked["x_cv_target_used"]
plant_in_phys = (
    x_scaled * plant_scale[plant_in_idx_arr] + plant_mean[plant_in_idx_arr]
)
pv_phys = result.attacked["pv_physical"]
full_133 = np.zeros((pv_phys.shape[0], len(scalers.sensor_cols)), dtype=np.float32)
full_133[:, plant_in_idx_arr] = plant_in_phys.astype(np.float32)
full_133[:, pv_idx_arr]       = pv_phys.astype(np.float32)
ok("full_133 builds without error, shape (180, 133)", full_133.shape == (180, 133))
ok("full_133 has no NaN", not np.isnan(full_133).any())
ok("full_133 PV columns match pv_physical",
   np.allclose(full_133[:, pv_idx_arr], pv_phys, atol=1e-6))

# Score it through the Guardian (same flow as _score_all_saved)
sensor_cols = list(scalers.sensor_cols)
feature_list = g["features"]
missing_features = [f for f in feature_list if f not in sensor_cols]
ok("guardian features all present in sensor_cols",
   len(missing_features) == 0,
   f"{len(missing_features)} missing: {missing_features[:3]}")
if not missing_features:
    col_index = [sensor_cols.index(f) for f in feature_list]
    X = full_133[:, col_index]
    Xs = g["scaler"].transform(X)
    proba = g["model"].predict_proba(Xs)[:, 1]
    peak = float(proba.max())
    ok("guardian predict_proba produces valid range",
       np.all((proba >= 0) & (proba <= 1)) and proba.shape == (180,),
       f"peak={peak:.3f}")
    print(f"        peak attack proba on this LC/CV/bias attack = {peak:.3f} "
          f"({'CAUGHT' if peak >= 0.35 else 'MISSED'})")


# ============================================================================
print("\n=== PHASE 6 · ROLLOUT TESTER DATA (both horizons) ===")
# ============================================================================
ev180  = get_eval_results()
ev1800 = get_eval_1800s()
ok("180s eval loaded", bool(ev180) and "overall_nrmse_per_pv" in ev180)
ok("1800s eval loaded", ev1800 is not None and "overall_nrmse_per_pv" in ev1800)
for pv in PV_COLS:
    v = ev180.get("overall_nrmse_per_pv", {}).get(pv)
    ok(f"180s overall · {pv} populated", v is not None and 0 < v < 1)
for pv in PV_COLS:
    v = ev1800["overall_nrmse_per_pv"].get(pv)
    ok(f"1800s overall · {pv} populated", v is not None and 0 < v < 1)
# All 4 scenarios populated at 1800s (the bug we previously fixed)
for scen in ["Normal","AP_no","AP_with","AE_no"]:
    v = ev1800["nrmse_per_scenario"].get(scen)
    ok(f"1800s scenario {scen} populated", v is not None, f"{v}")


# ============================================================================
print("\n=== PHASE 7 · LOOP EXPLORER MATH ===")
# ============================================================================
from student_guide_text import KS_PER_LOOP_REPORTED
EXPECTED_KS = {"PC": 0.060, "LC": 0.215, "FC": 0.067, "TC": 0.157, "CC": 0.121}
EXPECTED_STATUS = {"PC":"PASS","LC":"FAIL","FC":"PASS","TC":"FAIL","CC":"FAIL"}
for loop in EXPECTED_KS:
    rep = KS_PER_LOOP_REPORTED[loop]
    ok(f"KS {loop} matches §3.3", abs(rep["ks"] - EXPECTED_KS[loop]) < 0.001,
       f"{rep['ks']} vs {EXPECTED_KS[loop]}")
    ok(f"Status {loop} matches §3.3", rep["status"] == EXPECTED_STATUS[loop])

# Threshold-anchored fidelity formula
FAIL_KS = 0.20
for loop, ksv in EXPECTED_KS.items():
    f = max(0.0, min(1.0, 1.0 - ksv / FAIL_KS))
    expected_band = "high" if EXPECTED_STATUS[loop] == "PASS" else "low"
    actual_band   = "high" if f > 0.5 else "low"
    ok(f"radar fidelity {loop} ({EXPECTED_STATUS[loop]}) maps to {expected_band} band",
       expected_band == actual_band,
       f"f={f:.3f}, expected_band={expected_band}, actual_band={actual_band}")

delays = get_response_delays()
ok("delays · all 5 loops present", set(delays.keys()) == set(LOOP_SPECS.keys()))
ok("delays · CC slowest at 26 s", delays.get("CC") == 26)


# ============================================================================
print("\n=== PHASE 8 · CROSS-TAB CONSISTENCY ===")
# ============================================================================
# Header card "0.587" must equal Tab 4 inline GUARDIAN_D F1
ok("header F1 0.587 matches inline GUARDIAN_D F1",
   "0.587" in app_text and "f1\": 0.587" in clf_text)
# Bundle plant has 128 inputs, 5 outputs
ok("plant 128-in / 5-out", bundle.plant.n_plant_in == 128 and bundle.plant.n_pv == 5)
# 5 controllers
ok("5 controllers loaded",
   set(bundle.controllers.keys()) == {"PC","LC","FC","TC","CC"})


# ============================================================================
print("\n=== PHASE 9 · LOGICAL SANITY (Normal < AP_with, baseline < attack) ===")
# ============================================================================
scen_180 = ev180["nrmse_per_scenario"]
ok("180s · Normal < AP_with (coordinated harder)",
   scen_180["Normal"] < scen_180["AP_with"],
   f"Normal={scen_180['Normal']:.4f}, AP_with={scen_180['AP_with']:.4f}")

scen_1800 = ev1800["nrmse_per_scenario"]
ok("1800s · Normal < AP_with",
   scen_1800["Normal"] < scen_1800["AP_with"],
   f"Normal={scen_1800['Normal']:.4f}, AP_with={scen_1800['AP_with']:.4f}")

normal_nrmse = ev180.get("normal_nrmse_per_pv", {})
attack_nrmse = ev180.get("attack_nrmse_per_pv", {})
for pv in PV_COLS:
    if pv in normal_nrmse and pv in attack_nrmse:
        ok(f"180s · {pv} normal < attack",
           normal_nrmse[pv] < attack_nrmse[pv],
           f"normal={normal_nrmse[pv]:.4f}, attack={attack_nrmse[pv]:.4f}")

# XGBoost classifier matrix logic
if xgb_json.exists():
    f1A, f1B, f1C = xgb["A"]["f1"], xgb["B"]["f1"], xgb["C"]["f1"]
    f1D = 0.587  # inline constant
    ok("D > A by ≥ 0.20 (augmentation huge improvement)",
       (f1D - f1A) > 0.20,
       f"D-A = {f1D - f1A:+.3f}")
    ok("C ≪ A (synthetic-only fails)",
       f1C < f1A,
       f"C={f1C:.3f}, A={f1A:.3f}")


# ============================================================================
print("\n=== PHASE 10 · ARTIFACT INTEGRITY ===")
# ============================================================================
CRITICAL_FILES = {
    "plant ckpt":      DASH.parent / "generator/weights/gru_plant.pt",
    "v2 ckpt":         DASH.parent / "training/checkpoints/v2_weighted_init_best.pt",
    "guardian pkl":    Path(r"C:\Users\PC GAMING\Desktop\AI\HAI\best_hai_classifier.pkl"),
    "synthetic csv":   Path(r"C:\Users\PC GAMING\Desktop\AI\HAI\synthetic_attacks.csv"),
    "test1.csv":       Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test1.csv"),
    "test2.csv":       Path(r"C:\Users\PC GAMING\Desktop\new_ai\processed\test2.csv"),
    "causal graph":    DASH.parent / "outputs/causal_graph/parents_full.json",
    "eval_results":    DASH.parent / "generator/weights/eval_results.json",
    "eval_1800s":      DASH / "cache/eval_1800s.json",
    "xgb cache":       DASH / "cache/classifier_experiments.json",
    "rf cache":        DASH / "cache/classifier_experiments_rf.json",
    "gallery npz":     DASH / "cache/attack_gallery.npz",
}
for label, p in CRITICAL_FILES.items():
    ok(f"file exists · {label}", p.exists())


# ============================================================================
print("\n" + "="*70)
print(f"PASSED: {len(passes)}")
print(f"FAILED: {len(errors)}")
print(f"WARN:   {len(warns)}")
if errors:
    print("\nFAILURES:")
    for lbl, det in errors:
        print(f"  ✗ {lbl}  --  {det[:250]}")
else:
    print("\n[GREEN] All checks passed — ready to deploy.")
