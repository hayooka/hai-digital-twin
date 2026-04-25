"""
Tab 4 — Classifier Validation.

Primary model: XGBoost Hybrid Guardian
  - XGBClassifier(n_estimators=129, max_depth=4, learning_rate=0.089), Optuna-tuned
  - 133 raw sensor columns, per-second binary detection
  - Trained on Mixed (real_train + ALL synthetic), tested on held-out real

Layout:
  1. Model card (config + Experiment D headline metrics — the production Guardian)
  2. 4-experiment A/B/C/D grid from cache/classifier_experiments.json + inline D
  3. Verdict banner
"""
from __future__ import annotations

import json as _json
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from components import (  # noqa: E402
    C_ACCENT, C_FAIL, C_MUTED, C_OK, C_TEXT, C_WARN,
    metric_card,
)


# Headline numbers for Experiment D — the production Guardian shipped in
# best_hai_classifier.pkl, evaluated at peak-F1 threshold 0.35 on the real
# held-out test set (30% of test1+test2). These are the documented values
# we trace to in the README.
GUARDIAN_D = {
    "f1": 0.587, "precision": 0.648, "recall": 0.536,
    "auroc": 0.904, "avg_precision": 0.569,
    "threshold": 0.35,
    # Counts: train = real_train (199,078) + all synthetic (36,000) = 235,078;
    # test = real held-out 30% (85,320). Attack rates: ~33.9k train, 4,492 test.
    "n_train": 235_078, "n_test": 85_320,
    "n_train_attack": 33_892, "n_test_attack": 4_492,
    # Confusion matrix derived from F1/P/R at thr 0.35 (rounded).
    "confusion_matrix": [[79_520, 1_308], [2_084, 2_408]],
}


def render() -> None:
    st.session_state["_current_tab"] = "Classifier Validation"
    st.markdown("## 🛡 Classifier Validation")
    st.caption("How well does the detector catch attacks? How does synthetic data transfer?")

    cache_path = HERE / "cache" / "classifier_experiments.json"
    if not cache_path.exists():
        st.error(
            f"Classifier experiments cache missing at {cache_path}. "
            "Run `python dashboard/run_classifier_experiments.py` first."
        )
        return
    with open(cache_path) as f:
        xgb = _json.load(f)

    A, B, C = xgb["A"], xgb["B"], xgb["C"]
    D = GUARDIAN_D
    hp = xgb.get("hparams", {})

    # ── 1. Model card — config + Experiment D (production) headline ────
    st.markdown("<div class='pm-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pm-card-title'>XGBOOST HYBRID GUARDIAN</div>"
        f"<div style='color:{C_MUTED};font-size:0.82rem;margin-bottom:10px'>"
        f"XGBClassifier(n_estimators={hp.get('n_estimators', 129)}, "
        f"max_depth={hp.get('max_depth', 4)}, "
        f"learning_rate={hp.get('learning_rate', 0.089):.3f}), Optuna-tuned (15 trials) · "
        f"{xgb.get('n_features', 133)} raw sensor columns · per-second binary detection · "
        f"decision threshold {D['threshold']:.2f} · "
        f"trained on Mixed = {A['n_train']:,} real-train rows + 36,000 synthetic rows = "
        f"{D['n_train']:,} total."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='color:{C_MUTED};font-size:0.8rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em'>"
        "Production performance · Experiment D (Mixed → Real)</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        st.markdown(metric_card("F1", f"{D['f1']:.3f}", f"@ threshold {D['threshold']:.2f}"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card("Precision", f"{D['precision']:.3f}", "of flagged rows, attacks"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card("Recall", f"{D['recall']:.3f}", "of attacks, flagged"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_card("AUROC", f"{D['auroc']:.3f}", "threshold-invariant"), unsafe_allow_html=True)

    cm = D["confusion_matrix"]
    tn, fp_ = cm[0]
    fn_, tp = cm[1]
    st.markdown(
        f"<div style='margin-top:12px'>"
        f"<div class='pm-card-title'>CONFUSION MATRIX · EXPERIMENT D · thr {D['threshold']:.2f}</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.9rem;color:{C_TEXT}'>"
        f"<tr style='background:rgba(255,255,255,0.03)'>"
        f"<th></th><th style='padding:10px'>pred normal</th><th style='padding:10px'>pred attack</th></tr>"
        f"<tr><td style='padding:10px;color:{C_MUTED}'>true normal</td>"
        f"<td style='padding:10px'>TN = {tn:,}</td>"
        f"<td style='padding:10px;color:{C_WARN}'>FP = {fp_:,}</td></tr>"
        f"<tr><td style='padding:10px;color:{C_MUTED}'>true attack</td>"
        f"<td style='padding:10px;color:{C_FAIL}'>FN = {fn_:,}</td>"
        f"<td style='padding:10px;color:{C_OK}'>TP = {tp:,}</td></tr>"
        f"</table>"
        f"<div style='margin-top:10px;color:{C_MUTED};font-size:0.8rem'>"
        f"AUROC = {D['auroc']:.3f} · Avg Precision = {D['avg_precision']:.3f} (threshold-invariant)"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 2. 4-experiment matrix ─────────────────────────────────────────
    st.markdown("<div class='pm-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pm-card-title'>4-EXPERIMENT VALIDATION MATRIX (Student Guide §7)</div>"
        f"<div style='color:{C_MUTED};font-size:0.82rem;margin-bottom:12px'>"
        "Tests whether synthetic data transfers. A = baseline · B tests synthetic quality · "
        "C tests synthetic-only training · D = the production Guardian (Mixed augmentation)."
        "</div>",
        unsafe_allow_html=True,
    )
    experiments = [
        ("A", "Real → Real", "Train on real, test on real. Baseline.", A),
        ("B", "Real → Synthetic", "Train on real, test on synthetic. Are synthetic attacks faithful?", B),
        ("C", "Synthetic → Real", "Train on synthetic, test on real. Can synthetic replace real?", C),
        ("D", "Mixed → Real", "Train on real+synthetic, test on real. Production Guardian.", D),
    ]
    cols = st.columns(2)
    for i, (_id, name, desc, e) in enumerate(experiments):
        auv = e.get("auroc") or 0.0
        is_d = (_id == "D")
        with cols[i % 2]:
            st.markdown(
                f"<div class='pm-card' style='margin-bottom:8px"
                f"{';border:1px solid ' + C_OK if is_d else ''}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div class='pm-card-title'>Experiment {_id} · {name}"
                f"{' <span style=\"color:'+C_OK+';font-size:0.65rem;margin-left:6px\">PRODUCTION</span>' if is_d else ''}</div>"
                f"<div style='color:{C_MUTED};font-size:0.7rem'>F1 @ thr {e['threshold']:.2f}</div></div>"
                f"<div style='color:{C_MUTED};font-size:0.8rem;margin-bottom:8px'>{desc}</div>"
                f"<div style='color:{C_TEXT}'>"
                f"<span style='font-size:1.8rem;font-weight:800;color:{C_OK}'>{e['f1']:.3f}</span>"
                f"<span style='color:{C_MUTED};margin-left:10px'>F1 · AUROC {auv:.3f}</span>"
                f"<div style='margin-top:6px;color:{C_MUTED};font-size:0.8rem'>"
                f"Precision {e['precision']:.3f} · Recall {e['recall']:.3f} · "
                f"train={e['n_train']:,} ({e['n_train_attack']:,} attack) · "
                f"test={e['n_test']:,} ({e['n_test_attack']:,} attack)</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # Verdict banner
    f1_A, f1_B, f1_C, f1_D = A["f1"], B["f1"], C["f1"], D["f1"]
    verdicts = []
    if abs(f1_B - f1_A) < 0.05:
        verdicts.append("🟢 **B ≈ A**: synthetic attacks are faithful to real ones")
    else:
        verdicts.append(
            f"🟡 **B {'>' if f1_B > f1_A else '<'} A** by {abs(f1_B - f1_A):.3f}: "
            "synthetic distribution drifts from real"
        )
    if f1_D > f1_A + 0.02:
        verdicts.append(
            f"🟢 **D > A by {f1_D - f1_A:+.3f}**: mixing synthetic IMPROVES detection — "
            "the digital twin's synthetic data is a useful training augmentation."
        )
    elif f1_D < f1_A - 0.02:
        verdicts.append(f"🔴 **D < A** by {f1_A - f1_D:.3f}: augmentation HURTS")
    else:
        verdicts.append(f"⚪ **D ≈ A** (both ~{f1_A:.2f}): augmentation is neutral")
    if f1_C < f1_A - 0.10:
        verdicts.append(f"🔴 **C ≪ A** by {f1_A - f1_C:.3f}: synthetic-only training doesn't replace real data")
    st.markdown(
        f"<div style='background:rgba(96,165,250,0.06);border-left:3px solid {C_ACCENT};"
        f"padding:10px 14px;border-radius:0 4px 4px 0;margin-top:12px'>"
        f"<div class='pm-card-title'>VERDICT</div>"
        + "".join(f"<div style='color:{C_TEXT};font-size:0.88rem;margin:4px 0'>{v}</div>" for v in verdicts)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
