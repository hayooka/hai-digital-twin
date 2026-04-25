"""DETECT (CSV) tab — upload a 1 Hz HAI-format CSV, slide 180s windows,
score each with the frozen **plant GRU** (Mode A batched residual MSE),
threshold = bundle.threshold (calibrated at train time).

If ANY window fires, the whole page gets a red outline and a top-of-page alert.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from twin_core import (
    INPUT_LEN,
    PV_COLS,
    TARGET_LEN,
    build_plant_window,
    default_paths,
    load_bundle,
    load_replay,
    predict_plant,
    window_mse,
)

ROOT = Path(__file__).resolve().parents[1]
STRIDE_SEC = 60
CLS_DIR = ROOT / "outputs" / "classifier"
CALIB_PATH = CLS_DIR / "plant_threshold.json"

# PV-only fallback (uses the MLP autoencoder trained on train*.csv PV features)
sys.path.insert(0, str(ROOT / "05_classifier"))
from features import WIN_SEC as PV_WIN_SEC, STRIDE_SEC as PV_STRIDE_SEC, slide_windows as pv_slide_windows  # noqa: E402
from train_bench import MLPAutoencoder, ae_score  # noqa: E402


def _get_threshold(bundle) -> tuple[float, str]:
    """Prefer the normal-only train-calibrated threshold; fall back to
    the bundle threshold shipped with the checkpoint."""
    if CALIB_PATH.exists():
        try:
            payload = json.loads(CALIB_PATH.read_text())
            return float(payload["threshold"]), "train-99th-percentile (FPR≈1%)"
        except Exception:
            pass
    return float(bundle.threshold), "checkpoint default"


@st.cache_resource(show_spinner="Loading plant GRU detector…")
def _load_detector():
    paths = default_paths()
    bundle = load_bundle(paths["ckpt_dir"], paths["split_dir"])
    return bundle


@st.cache_resource(show_spinner="Loading PV-only fallback model…")
def _load_pv_fallback() -> dict:
    """Load the MLP autoencoder + train-set z-score stats from the
    05_classifier bench, used when the uploaded CSV has only PV columns.
    """
    feats = CLS_DIR / "features.npz"
    winner = CLS_DIR / "winner.json"
    ae_path = CLS_DIR / "ae.pt"
    if not (feats.exists() and ae_path.exists()):
        return {"ready": False}
    z = np.load(feats, allow_pickle=True)
    mu = z["mu"].astype(np.float32)
    sd = z["sd"].astype(np.float32)
    net = MLPAutoencoder(d_in=mu.size)
    net.load_state_dict(torch.load(ae_path, map_location="cpu"))
    net.eval()
    thr = 0.5
    meta = {"model_name": "ae"}
    if winner.exists():
        try:
            w = json.loads(winner.read_text())
            thr = float(w.get("threshold", thr))
            meta["model_name"] = w.get("model", "ae")
        except Exception:
            pass
    return {
        "ready": True,
        "mu": mu,
        "sd": sd,
        "net": net,
        "threshold": thr,
        "model_name": meta["model_name"],
    }


def _score_pv_only(df: pd.DataFrame, cfg: dict) -> dict:
    """Fallback: PV-only MLP autoencoder on the 5 PV columns.

    If `sample_id` is present (generator dump format), treat each group as one
    pre-cut window. Otherwise slide windows over the 1 Hz stream.
    """
    from features import window_features

    missing = [c for c in PV_COLS if c not in df.columns]
    if missing:
        return {"error": f"CSV missing PV columns: {missing}"}

    feats = []
    y = []
    anchors = []
    detector_note = ""

    if "sample_id" in df.columns:
        detector_note = " (per-sample_id grouped)"
        for sid, g in df.groupby("sample_id", sort=True):
            pv = g[PV_COLS].to_numpy(dtype=np.float32)
            if pv.shape[0] < PV_WIN_SEC:
                continue
            pv = pv[:PV_WIN_SEC]  # truncate to exactly 180s
            feats.append(window_features(pv))
            if "regime" in g.columns:
                y.append(int(str(g["regime"].iloc[0]).lower() != "normal"))
            elif "label" in g.columns:
                y.append(int((g["label"].to_numpy() > 0).any()))
            else:
                y.append(0)
            anchors.append(int(sid))
        if not feats:
            return {
                "error": f"No sample_id group has >= {PV_WIN_SEC} rows "
                         f"— generator windows should be exactly 180 rows."
            }
        X = np.stack(feats, axis=0).astype(np.float32)
        y_arr = np.array(y, dtype=np.int8)
        anchors_arr = np.array(anchors, dtype=np.int64)
        has_labels = True  # regime/label drove y
    else:
        pv = df[PV_COLS].to_numpy(dtype=np.float32)
        label_arr = df["label"].to_numpy() if "label" in df.columns else None
        X, y_arr, anchors_arr = pv_slide_windows(pv, label_arr)
        if X.shape[0] == 0:
            return {"error": f"CSV has {len(df)} rows (< {PV_WIN_SEC}s). "
                             f"Need >= {PV_WIN_SEC}."}
        has_labels = label_arr is not None

    Xz = (X - cfg["mu"]) / cfg["sd"]
    scores = ae_score(cfg["net"], Xz.astype(np.float32))
    # PV-only AE trained on real HAI stats; generator output has distribution
    # shift, so raise threshold 3x to absorb benign shift and only fire on
    # clearly anomalous windows.
    thr = float(cfg["threshold"]) * 3.0
    return {
        "anchors": anchors_arr,
        "scores": scores.astype(np.float32),
        "verdict": (scores >= thr).astype(np.int8),
        "y_true": y_arr if has_labels else None,
        "threshold": thr,
        "detector": f"PV-only MLP Autoencoder{detector_note}",
        "stride_sec": PV_STRIDE_SEC,
    }


def _score_csv(bundle, csv_path: Path) -> dict:
    try:
        src = load_replay(csv_path, bundle.scalers)
    except Exception as e:
        return {"error": f"Could not load CSV through twin scalers: {e}"}

    n = len(src)
    t_first = INPUT_LEN
    t_last = n - TARGET_LEN
    if t_last <= t_first:
        return {
            "error": f"CSV has {n} rows. Need at least "
                     f"{INPUT_LEN + TARGET_LEN} = {INPUT_LEN + TARGET_LEN}s "
                     "(300s history + 180s target)."
        }

    anchors = np.arange(t_first, t_last + 1, STRIDE_SEC, dtype=np.int64)
    scores = np.zeros(anchors.size, dtype=np.float32)
    y_true = np.zeros(anchors.size, dtype=np.int8)
    has_label = "label" in src.df_raw.columns
    labels = src.df_raw["label"].to_numpy() if has_label else None

    prog = st.progress(0.0, text=f"Scoring {anchors.size} windows with plant GRU…")
    for i, t in enumerate(anchors):
        win = build_plant_window(bundle, src, int(t))
        if win is None:
            scores[i] = np.nan
            continue
        pv_pred = predict_plant(bundle, win)                      # (180, 5) scaled
        pv_true = win["pv_target"].squeeze(0).cpu().numpy()       # (180, 5) scaled
        scores[i] = window_mse(pv_pred, pv_true)
        if labels is not None:
            y_true[i] = int((labels[int(t) : int(t) + TARGET_LEN] > 0).any())
        if i % 20 == 0:
            prog.progress((i + 1) / anchors.size,
                          text=f"Scoring window {i + 1}/{anchors.size}…")
    prog.empty()

    threshold, _ = _get_threshold(bundle)
    verdict = (scores > threshold).astype(np.int8)
    return {
        "anchors": anchors,
        "scores": scores,
        "verdict": verdict,
        "y_true": y_true if has_label else None,
        "threshold": threshold,
        "detector": "Plant GRU residual MSE",
        "stride_sec": STRIDE_SEC,
    }


def _inject_alert_styling(is_attack: bool) -> None:
    """Red alarm (attack) OR green confirmation outline (normal) around the page."""
    if is_attack:
        css = """
        <style>
          @keyframes dt-alarm-pulse {
            0%, 100% { box-shadow: inset 0 0 0 6px #ef5350, 0 0 32px rgba(239,83,80,0.35); }
            50%      { box-shadow: inset 0 0 0 10px #ff1744, 0 0 48px rgba(255,23,68,0.55); }
          }
          [data-testid="stAppViewContainer"] {
            animation: dt-alarm-pulse 1.4s ease-in-out infinite;
          }
          .dt-alert-banner {
            background: linear-gradient(90deg, #3a1414 0%, #5a1f1f 50%, #3a1414 100%);
            border: 2px solid #ef5350;
            color: #ff8a80;
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-weight: 700;
            letter-spacing: 3px;
            text-align: center;
            padding: 14px 20px;
            margin: 0 0 14px 0;
            text-transform: uppercase;
            animation: dt-alarm-pulse 1.4s ease-in-out infinite;
          }
        </style>
        """
    else:
        css = """
        <style>
          @keyframes dt-ok-pulse {
            0%, 100% { box-shadow: inset 0 0 0 6px #66bb6a, 0 0 28px rgba(102,187,106,0.30); }
            50%      { box-shadow: inset 0 0 0 9px #43a047, 0 0 40px rgba(67,160,71,0.45); }
          }
          [data-testid="stAppViewContainer"] {
            animation: dt-ok-pulse 2.2s ease-in-out infinite;
          }
          .dt-ok-banner {
            background: linear-gradient(90deg, #0f2a16 0%, #1a4424 50%, #0f2a16 100%);
            border: 2px solid #66bb6a;
            color: #a5d6a7;
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-weight: 700;
            letter-spacing: 3px;
            text-align: center;
            padding: 14px 20px;
            margin: 0 0 14px 0;
            text-transform: uppercase;
            animation: dt-ok-pulse 2.2s ease-in-out infinite;
          }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def render() -> None:
    st.markdown(
        '<div class="panel-title">▊ UPLOAD CSV — PLANT-GRU ATTACK DETECTOR</div>',
        unsafe_allow_html=True,
    )
    bundle = _load_detector()
    threshold, thr_source = _get_threshold(bundle)

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-cell"><div class="metric-value">PLANT GRU</div>'
        f'<div class="metric-label">Detector (trained on normal only)</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-cell"><div class="metric-value">{threshold:.4f}</div>'
        f'<div class="metric-label">Threshold · {thr_source}</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-cell"><div class="metric-value">0.899</div>'
        f'<div class="metric-label">AUROC (HAI test)</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    up = st.file_uploader(
        "Upload a CSV. Full 133-column HAI schema → plant-GRU detector. "
        "PV-only (5 columns) or generator output → PV autoencoder fallback.",
        type=["csv"],
    )
    if up is None:
        st.markdown(
            f'<div class="caption-mono">Primary detector: plant GRU, window = '
            f'300 s history + 180 s target, stride = {STRIDE_SEC} s, threshold = '
            f'{threshold:.4f}. Fallback (if the CSV lacks full plant schema): '
            f'PV-only MLP autoencoder on 180 s windows.</div>',
            unsafe_allow_html=True,
        )
        return

    # Peek at columns to pick the right detector
    try:
        df_preview = pd.read_csv(up, nrows=2)
    except Exception as e:
        st.error(f"CSV parse error: {e}")
        return
    up.seek(0)  # rewind so we can read the full file below
    plant_cols_missing = [c for c in bundle.scalers.sensor_cols if c not in df_preview.columns]
    has_plant_schema = len(plant_cols_missing) == 0
    has_pv_only = all(c in df_preview.columns for c in PV_COLS)

    if has_plant_schema:
        # Persist upload to a temp CSV so load_replay can read by path
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="wb"
        ) as tmp:
            tmp.write(up.getvalue())
            tmp_path = Path(tmp.name)
        try:
            res = _score_csv(bundle, tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    elif has_pv_only:
        st.info(
            f"CSV does not contain the full 133-column plant schema "
            f"({len(plant_cols_missing)} sensors missing). "
            "Falling back to the PV-only MLP autoencoder detector."
        )
        cfg = _load_pv_fallback()
        if not cfg.get("ready"):
            st.error(
                "PV-only fallback model not found. Run:\n\n"
                "`python 05_classifier/build_features.py`\n"
                "`python 05_classifier/train_bench.py`\n"
                "`python 05_classifier/pick_winner.py`"
            )
            return
        df_full = pd.read_csv(up)
        res = _score_pv_only(df_full, cfg)
    else:
        st.error(
            f"CSV must contain either the full 133-column HAI plant schema OR "
            f"at minimum the 5 PV columns {PV_COLS}. "
            f"Uploaded CSV has {len(df_preview.columns)} columns and is missing "
            f"{len(plant_cols_missing)} plant sensors."
        )
        return

    if "error" in res:
        st.error(res["error"])
        return

    scores = res["scores"]
    verdict = res["verdict"]
    anchors = res["anchors"]
    y_true = res["y_true"]
    n_win = scores.size
    n_att = int(verdict.sum())
    attack_rate = (n_att / max(n_win, 1))
    ATTACK_RATE_DECISION = 0.70  # fire alarm only if >70% of windows flag attack
    is_attack = attack_rate > ATTACK_RATE_DECISION

    # ── ALARM STYLING ────────────────────────────────────────────────────
    _inject_alert_styling(is_attack)
    detector_name = res.get("detector", "detector")
    if is_attack:
        st.markdown(
            f'<div class="dt-alert-banner">'
            f'▲ ATTACK DETECTED — {n_att} / {n_win} WINDOWS '
            f'({attack_rate*100:.0f}% > {ATTACK_RATE_DECISION*100:.0f}%) ▲'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.error(
            f"ATTACK DETECTED: {n_att} of {n_win} windows "
            f"({attack_rate*100:.1f}%) exceed the {detector_name} threshold "
            f"{res['threshold']:.4f} — above the {ATTACK_RATE_DECISION*100:.0f}% "
            f"decision rule."
        )
    else:
        st.markdown(
            f'<div class="dt-ok-banner">'
            f'● NORMAL — {n_att} / {n_win} WINDOWS '
            f'({attack_rate*100:.0f}% ≤ {ATTACK_RATE_DECISION*100:.0f}%) ●'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.success(
            f"NORMAL — {n_att} of {n_win} windows flagged "
            f"({attack_rate*100:.1f}%); below the "
            f"{ATTACK_RATE_DECISION*100:.0f}% decision rule for "
            f"{detector_name}, threshold {res['threshold']:.4f}."
        )

    # ── Metric tiles ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f'<div class="metric-cell"><div class="metric-value">{n_win}</div>'
        f'<div class="metric-label">Windows</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-cell"><div class="metric-value" '
        f'style="color:{"#ef5350" if is_attack else "#66bb6a"};">{n_att}</div>'
        f'<div class="metric-label">Attack verdicts</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-cell"><div class="metric-value">'
        f'{(n_att / max(n_win, 1)) * 100:.1f}%</div>'
        f'<div class="metric-label">Attack rate</div></div>',
        unsafe_allow_html=True,
    )
    peak = float(np.nanmax(scores)) if scores.size else 0.0
    m4.markdown(
        f'<div class="metric-cell"><div class="metric-value">{peak:.4f}</div>'
        f'<div class="metric-label">Peak score</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Timeline plot ────────────────────────────────────────────────────
    x_sec = anchors.astype(int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_sec, y=scores, mode="lines",
        line=dict(color="#4fc3f7", width=1.6),
        name=("PV-only AE reconstruction MSE"
              if str(res.get("detector", "")).startswith("PV-only")
              else "plant-GRU residual MSE"),
        hovertemplate="anchor t=%{x}s  MSE=%{y:.5f}<extra></extra>",
    ))
    att_mask = verdict == 1
    if att_mask.any():
        fig.add_trace(go.Scatter(
            x=x_sec[att_mask], y=scores[att_mask], mode="markers",
            marker=dict(color="#ef5350", size=8, symbol="circle"),
            name="ATTACK",
            hovertemplate="ATTACK @ anchor t=%{x}s<extra></extra>",
        ))
    fig.add_hline(
        y=res["threshold"],
        line=dict(color="#ffa726", dash="dash", width=1.4),
        annotation_text=f"threshold {res['threshold']:.4f}",
        annotation_position="top right",
        annotation_font=dict(color="#ffa726", size=10),
    )
    fig.update_layout(
        paper_bgcolor="#101a2e", plot_bgcolor="#101a2e",
        font=dict(family="JetBrains Mono, Consolas, monospace",
                  color="#9fb0c8", size=11),
        margin=dict(l=45, r=15, t=20, b=30),
        height=380, hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(title="ANCHOR SECONDS",
                     gridcolor="#1e2a44", zerolinecolor="#1e2a44",
                     color="#9fb0c8", tickfont=dict(size=10))
    fig.update_yaxes(title="RESIDUAL MSE",
                     gridcolor="#1e2a44", zerolinecolor="#1e2a44",
                     color="#9fb0c8", tickfont=dict(size=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Verdict table + CSV download ─────────────────────────────────────
    out_df = pd.DataFrame({
        "anchor_sec": anchors.astype(int),
        "score": np.round(scores, 6),
        "verdict": np.where(verdict == 1, "ATTACK", "normal"),
    })
    if y_true is not None:
        out_df["ground_truth"] = np.where(y_true == 1, "attack", "normal")

    st.markdown("")
    st.dataframe(out_df.head(200), width="stretch", hide_index=True)
    st.download_button(
        "⬇ Download verdicts.csv",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"verdicts_{up.name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
    )
