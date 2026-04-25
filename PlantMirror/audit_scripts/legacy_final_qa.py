"""
Final QA — programmatic exercise of every widget + numerical audit.
"""
import sys, json, traceback, os, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

DASH = Path(r"C:\Users\PC GAMING\Desktop\new_ai\hai-digital-twin\dashboard")
sys.path.insert(0, str(DASH))

os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
import streamlit as st
st.cache_resource = lambda *a, **k: (lambda f: f) if not a or not callable(a[0]) else a[0]
st.cache_data     = lambda *a, **k: (lambda f: f) if not a or not callable(a[0]) else a[0]

errors = []
passes = []
def ok(label, cond, detail=""):
    (passes if cond else errors).append((label, detail))
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}{'  -- ' + detail if detail else ''}")

# ============================================================================
# PHASE 1 — Data-loader contracts
# ============================================================================
print("\n=== PHASE 1 · DATA LOADERS ===")
from data_loader import (
    get_bundle, get_replay, get_guardian, get_guardian_scores,
    get_causal_graph, get_coupling_matrix, get_response_delays,
    get_eval_results, get_bench_json, get_attack_gallery,
    LOOP_SPECS, PV_COLS,
)

bundle = get_bundle()
ok("plant loads · 128-in, 5-out", bundle.plant.n_plant_in == 128 and bundle.plant.n_pv == 5)
ok("5 controllers", set(bundle.controllers.keys()) == {"PC","LC","FC","TC","CC"})
src = get_replay()
ok("replay loaded", len(src) > 1000, f"{len(src)} rows")
g = get_guardian()
ok("guardian XGBoost pickle loads", g["available"] and len(g["features"]) == 133)
scores = get_guardian_scores()
ok("guardian proba[] == y_true[] length", len(scores["proba"]) == len(scores["y_true"]))
ok("test set 284,398 rows", scores["n_rows"] == 284398)
graph = get_causal_graph()
ok("causal graph non-empty", len(graph) > 10)
cm = get_coupling_matrix()
mat = np.array(cm["matrix"])
ok("coupling matrix 5x5", mat.shape == (5, 5))
ok("coupling FC->CC aliased (was 0 before fix)", mat[cm["loops"].index("FC"), cm["loops"].index("CC")] > 0)
delays = get_response_delays()
ok("delays for all 5 loops", set(delays.keys()) == {"PC","LC","FC","TC","CC"})
ok("CC is slowest at 26 s", max(delays, key=delays.get) == "CC" and delays["CC"] == 26)
ev = get_eval_results()
ok("eval_results has val_loss 0.00031", abs(ev["val_loss"] - 0.000310) < 1e-5)
ok("eval_results has 4 scenarios",
   set(ev["nrmse_per_scenario"].keys()) == {"Normal","AP_no","AP_with","AE_no"})
bench = get_bench_json()
ok("bench.json has 3 baselines", len(bench["results"]) == 3)
gal = get_attack_gallery()
ok("gallery 30 attacks · (30,180,5)", gal is not None and gal["baseline_pv"].shape == (30, 180, 5))

# ============================================================================
# PHASE 2 — Numerical source-of-truth checks
# ============================================================================
print("\n=== PHASE 2 · NUMERICAL SOURCE OF TRUTH ===")
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
proba = scores["proba"]; y_true = scores["y_true"]

auroc = roc_auc_score(y_true, proba)
ap = average_precision_score(y_true, proba)
ok("Guardian AUROC ~0.904", abs(auroc - 0.904) < 0.005, f"{auroc:.4f}")
ok("Guardian AP ~0.569", abs(ap - 0.569) < 0.005, f"{ap:.4f}")

# Threshold-specific F1/P/R at the 3 preset points
for thr, exp_f1 in [(0.35, 0.581), (0.50, 0.559), (0.60, 0.511)]:
    y_pred = (proba >= thr).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ok(f"Guardian F1 @ thr {thr}", abs(f1 - exp_f1) < 0.005, f"{f1:.4f} vs exp {exp_f1}")

# Classifier experiments cache
for path in ["classifier_experiments.json", "classifier_experiments_rf.json"]:
    p = DASH / "cache" / path
    ok(f"{path} exists", p.exists())
    if p.exists():
        with open(p) as f: d = json.load(f)
        ok(f"{path} has A/B/C", all(k in d for k in ["A","B","C"]))
        for k in ["A","B","C"]:
            ok(f"{path}[{k}] has F1+P+R", all(m in d[k] for m in ["f1","precision","recall"]))

# Eval results per-PV per-scenario completeness
for pv in PV_COLS:
    per_scen = ev["nrmse_per_pv_per_scenario"].get(pv, {})
    ok(f"eval_results[{pv}] has all 4 scenarios", set(per_scen.keys()) == {"Normal","AP_no","AP_with","AE_no"})

# Attack engine integrity
from data_loader import AttackSpec, InjectionPoint, AttackType, run_attack_sim
for ip, so, exp_visible in [
    (InjectionPoint.SP, -30, True),
    (InjectionPoint.CV, 0,   True),
    (InjectionPoint.PV, -30, True),
]:
    spec = AttackSpec(target_loop="LC", injection_point=ip, attack_type=AttackType.BIAS,
                      start_offset=so, duration=60, magnitude=20.0)
    r = run_attack_sim(bundle, src, t_end=1000, spec=spec, scenario=0)
    dpv = float(np.max(np.abs(r.attacked["pv_physical"] - r.baseline["pv_physical"])))
    ok(f"attack LC/{ip.value}/bias start={so} produces drift", dpv > 0.1, f"dPV={dpv:.3f}")

# ============================================================================
# PHASE 3 — Programmatic widget exercise (Streamlit AppTest)
# ============================================================================
print("\n=== PHASE 3 · APPTEST (every widget interaction) ===")
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(DASH / "app.py"), default_timeout=180)
at.run()
ok("app initial render · no exceptions", not at.exception, str(at.exception)[:300] if at.exception else "")
ok("tabs >= 4 visible", len(at.tabs) >= 4, f"found {len(at.tabs)}")

# Loop selector across all 5 loops
for loop in ["PC","LC","FC","TC","CC"]:
    try:
        matches = [r for r in at.radio if r.key == "loop_sel"]
        if matches:
            matches[0].set_value(loop).run()
            ok(f"loop selector → {loop}", not at.exception)
    except Exception as e:
        errors.append((f"loop_sel={loop}", str(e)))

# Guardian threshold slider
for thr in [0.10, 0.35, 0.50, 0.60, 0.90]:
    try:
        matches = [s for s in at.slider if s.key == "guardian_threshold"]
        if matches:
            matches[0].set_value(thr).run()
            ok(f"threshold={thr}", not at.exception)
    except Exception as e:
        errors.append((f"threshold={thr}", str(e)))

# Gallery filters
try:
    for m in [m for m in at.multiselect if m.key in ("gal_loop","gal_ip","gal_type")]:
        m.set_value([m.options[0]]).run()
        ok(f"gallery filter {m.key} single", not at.exception)
except Exception as e:
    errors.append(("gallery filters", str(e)))

# Rollout scenario toggle
try:
    for r in [r for r in at.radio if r.key == "rollout_scen"]:
        for sc in ["Overall","Normal","AP_no","AP_with","AE_no"]:
            r.set_value(sc).run()
            ok(f"rollout scenario={sc}", not at.exception)
except Exception as e:
    errors.append(("rollout_scen", str(e)))

# Custom attack: pick each injection × each type
try:
    for ip in ["SP","CV","PV"]:
        for t in ["bias","freeze","replay"]:
            [s for s in at.selectbox if s.key == "cus_ip"][0].set_value(ip).run()
            [s for s in at.selectbox if s.key == "cus_at"][0].set_value(t).run()
            run_btn = [b for b in at.button if b.key == "cus_run"]
            if run_btn:
                run_btn[0].click().run()
                ok(f"custom attack {ip}/{t} runs", not at.exception)
except Exception as e:
    errors.append(("custom attack sweep", str(e)[:200]))

# ============================================================================
# PHASE 4 — Streamlit runtime log scan
# ============================================================================
print("\n=== PHASE 4 · RUNTIME LOG SCAN ===")
log_path = Path("/tmp/streamlit_final.log")
if log_path.exists():
    text = log_path.read_text(encoding="utf-8", errors="replace")
    n_traceback = text.count("Traceback")
    n_error = sum(1 for line in text.splitlines()
                  if any(k in line for k in ["StreamlitAPIException","RuntimeError","ValueError","KeyError"]))
    ok("no Traceback in runtime log", n_traceback == 0, f"found {n_traceback}")
    ok("no Streamlit API exceptions", n_error == 0, f"found {n_error}")

# ============================================================================
# FINAL TALLY
# ============================================================================
print("\n" + "=" * 70)
print(f"PASSED: {len(passes)}")
print(f"FAILED: {len(errors)}")
if errors:
    print("\nFAILURES:")
    for lbl, det in errors:
        print(f"  * {lbl}  --  {det[:250]}")
else:
    print("\nALL CHECKS PASSED. Dashboard is ready to share.")
