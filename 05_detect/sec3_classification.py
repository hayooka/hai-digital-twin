"""
sec3_classification.py — Section 3: Synthetic Data Accuracy & Classifier

Proves two things:
  1. Generated data is accurate — synthetic PV distributions match real per scenario
  2. Generated data is useful  — classifier trained on synthetic only generalises to real

Experiments:
  A. Baseline : train on imbalanced real data,  test on real  → shows problem
  B. TRTS     : train on synthetic data only,   test on real  → proves concept
  C. Mixed    : train on real Normal + synthetic attacks, test on real → best result

Produces:
  figures/s3c_data_quality.png     — real vs synthetic PV distributions per scenario
  figures/s3c_experiment_bar.png   — macro F1 comparison across 3 experiments
  figures/s3c_confusion_trts.png   — confusion matrix: TRTS experiment
  figures/s3c_confusion_mixed.png  — confusion matrix: Mixed experiment

Usage:
    python report_plots/code/sec3_classification.py
"""

import sys, json
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                              recall_score, classification_report)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 128
CKPT_DIR   = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"
OUT_DIR    = ROOT / "report_plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CTRL_LOOPS     = ["PC", "LC", "FC", "TC", "CC"]
CTRL_H         = {"PC": 64, "LC": 64, "FC": 128, "TC": 64, "CC": 64}
N_SYNTH        = 3292   # match Normal count — fully balanced
SCENARIO_NAMES = {0: "Normal", 1: "AP\_no", 2: "AP\_with", 3: "AE\_no"}
SC_LABELS      = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SC_COLORS      = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}
PV_SHORT       = [c.replace("P1_", "") for c in PV_COLS]

EXTRA_CHANNELS = {
    "PC": ["P1_PCV02D", "P1_FT01",   "P1_TIT01"],
    "LC": ["P1_FT03",   "P1_FCV03D", "P1_PCV01D"],
    "FC": ["P1_PIT01",  "P1_LIT01",  "P1_TIT03"],
    "TC": ["P1_FT02",   "P1_PIT02",  "P1_TIT02"],
    "CC": ["P1_PP04D",  "P1_FCV03D", "P1_PCV02D"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_models(data):
    cd   = data["ctrl"]
    TL   = data["metadata"]["target_len"]
    ckpt = torch.load(CKPT_DIR / "gru_plant.pt", map_location=DEVICE)
    ms   = ckpt["model_state"]
    emb  = ms["scenario_emb.weight"].shape[1]
    plant = GRUPlant(
        n_plant_in  = ckpt.get("n_plant_in", ms["encoder.weight_ih_l0"].shape[1] - emb),
        n_pv        = ckpt.get("n_pv",       ms["fc.3.weight"].shape[0]),
        hidden      = ckpt["hidden"],
        layers      = ckpt["layers"],
        n_scenarios = ms["scenario_emb.weight"].shape[0],
        n_haiend    = ckpt.get("n_haiend", 0),
    ).to(DEVICE)
    plant.load_state_dict(ms, strict=False)
    plant.eval()
    ctrls = {}
    for ln in CTRL_LOOPS:
        n_in = cd[ln]["X_train"].shape[-1]
        m = (CCSequenceModel(n_inputs=n_in, hidden=CTRL_H[ln], layers=2,
                             dropout=0.0, output_len=TL)
             if ln == "CC" else
             GRUController(n_inputs=n_in, hidden=CTRL_H[ln], layers=2,
                           dropout=0.0, output_len=TL)).to(DEVICE)
        p = CKPT_DIR / f"gru_ctrl_{ln.lower()}.pt"
        if p.exists():
            c = torch.load(p, map_location=DEVICE)
            m.load_state_dict(c["model_state"], strict=False)
        m.eval()
        ctrls[ln] = m
    return plant, ctrls, TL


def generate_scenario(plant, ctrls, ctrl_cv_idx, plant_data, ctrl_data,
                       sc_id, n_target, TL, NP):
    sc_arr  = plant_data["scenario_test"]
    mask    = np.where(sc_arr == sc_id)[0]
    rep     = np.random.choice(mask, size=n_target, replace=True)
    out     = np.zeros((n_target, TL, NP), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_target, BATCH):
            sl  = slice(i, i + BATCH)
            idx = rep[sl]
            xb  = torch.tensor(plant_data["X_test"][idx]).float().to(DEVICE)
            xcb = torch.tensor(plant_data["X_cv_target_test"][idx]).float().to(DEVICE).clone()
            pb  = torch.tensor(plant_data["pv_init_test"][idx]).float().to(DEVICE)
            sb  = torch.full((xb.size(0),), sc_id, dtype=torch.long, device=DEVICE)
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_idx: continue
                cp = ctrls[ln].predict(
                    torch.tensor(ctrl_data[ln]["X_test"][idx]).float().to(DEVICE),
                    target_len=TL)
                xcb[:, :, ctrl_cv_idx[ln]:ctrl_cv_idx[ln]+1] = cp
            pv, _ = plant.predict(xb, xcb, pb, sb)
            out[i:i + xb.size(0)] = pv.cpu().numpy()
    return out


def extract_features(traj):
    """(N, T, K) → (N, K*6) statistical features."""
    feats = []
    for k in range(traj.shape[-1]):
        r = traj[:, :, k]
        feats += [r.mean(1), r.std(1), r.min(1), r.max(1),
                  np.abs(r).mean(1), np.diff(r, axis=1).mean(1)]
    return np.stack(feats, axis=1)


def run_clf(X_tr, y_tr, X_te, y_te, label):
    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    clf  = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)
    macro  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"\n  [{label}]  Macro F1 = {macro:.4f}")
    print(classification_report(y_te, y_pred,
          target_names=[SC_LABELS[i] for i in range(4)], zero_division=0))
    return y_pred, y_prob, macro, (clf, sc)


# ── Plot 1: Data quality — real vs synthetic PV distributions ─────────────────

def plot_data_quality(real_trajs, synth_trajs):
    """
    For each attack scenario, compare the mean PV trajectory of real vs synthetic.
    One row per scenario (AP_no, AP_with, AE_no), one column per PV.
    """
    attack_ids = [1, 2, 3]
    fig, axes  = plt.subplots(len(attack_ids), len(PV_COLS),
                               figsize=(16, 3.5 * len(attack_ids)),
                               sharex=True)

    for row, sc_id in enumerate(attack_ids):
        real   = real_trajs[sc_id]    # (N_real, TL, 5)
        synth  = synth_trajs[sc_id]   # (N_synth, TL, 5)
        color  = SC_COLORS[sc_id]
        t      = np.arange(real.shape[1])

        for col in range(len(PV_COLS)):
            ax = axes[row, col]
            r_mean = real[:, :, col].mean(0)
            r_std  = real[:, :, col].std(0)
            s_mean = synth[:, :, col].mean(0)
            s_std  = synth[:, :, col].std(0)

            ax.plot(t, r_mean, color="black",  lw=1.8, label="Real mean")
            ax.fill_between(t, r_mean-r_std, r_mean+r_std,
                            color="black", alpha=0.10)
            ax.plot(t, s_mean, color=color, lw=1.8,
                    linestyle="--", label="Synthetic mean")
            ax.fill_between(t, s_mean-s_std, s_mean+s_std,
                            color=color, alpha=0.15)

            ax.grid(True, linestyle="--", alpha=0.3)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(PV_SHORT[col], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(SC_LABELS[sc_id], fontsize=10,
                              fontweight="bold", color=color)

    # shared legend
    handles = [
        plt.Line2D([0],[0], color="black", lw=1.8, label="Real (mean ± std)"),
        plt.Line2D([0],[0], color="gray",  lw=1.8, linestyle="--",
                   label="Synthetic (mean ± std)"),
    ]
    fig.legend(handles=handles, fontsize=10, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Generated vs Real PV Trajectories — Mean ± Std per Scenario",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = OUT_DIR / "s3c_data_quality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 2: Experiment comparison bar chart ───────────────────────────────────

def plot_experiment_bar(results):
    """
    results = dict: experiment_name → {"macro_f1": float, "per_class": list}
    """
    names   = list(results.keys())
    macros  = [results[n]["macro_f1"] for n in names]
    colors  = ["#90A4AE", "#FF5722", "#4CAF50"]
    descs   = [
        "Imbalanced real\ndata only",
        "Synthetic data\nonly (TRTS)",
        "Real Normal +\nSynthetic attacks",
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, macros, color=colors, alpha=0.88, width=0.5)
    for bar, v in zip(bars, macros):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    for bar, desc in zip(bars, descs):
        ax.text(bar.get_x() + bar.get_width()/2, -0.10, desc,
                ha="center", va="top", fontsize=8, color="gray",
                transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro F1 Score (4-class)", fontsize=12)
    ax.set_title("Attack Classification: Real vs Synthetic Training Data",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1.2, label="Random baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = OUT_DIR / "s3c_experiment_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 3: Confusion matrix ──────────────────────────────────────────────────

def plot_confusion(y_true, y_pred, title, fname):
    labels  = [SC_LABELS[i] for i in range(4)]
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, subtitle, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised"],
        [".0f", ".2f"]
    ):
        im = ax.imshow(data, cmap="Blues", interpolation="nearest")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=30,
                                                      ha="right", fontsize=10)
        ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        thresh = data.max() / 2.0
        for i in range(4):
            for j in range(4):
                ax.text(j, i, format(data[i,j], fmt),
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if data[i,j] > thresh else "black")

    fig.suptitle(f"{title}  —  Macro F1 = {macro:.3f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data        = load_and_prepare_data()
    plant_data  = data["plant"]
    ctrl_data   = data["ctrl"]
    sensor_cols = data["metadata"]["sensor_cols"]
    TL          = data["metadata"]["target_len"]
    NP          = plant_data["n_pv"]
    sc_arr      = plant_data["scenario_test"]   # ground truth labels

    # augment controller inputs
    plant_scaler = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    col_idx = {c: i for i, c in enumerate(sensor_cols)}
    for ln, extra_cols in EXTRA_CHANNELS.items():
        for ec in extra_cols:
            if ec not in col_idx: continue
            ei = col_idx[ec]
            mean_e, scale_e = plant_scaler.mean_[ei], plant_scaler.scale_[ei]
            for split in ("train", "val", "test"):
                npz = np.load(f"{PROCESSED_DATA_DIR}/{split}_data.npz")
                raw = npz["X"][:, :, [ei]].astype(np.float32)
                ctrl_data[ln][f"X_{split}"] = np.concatenate(
                    [ctrl_data[ln][f"X_{split}"], (raw - mean_e) / scale_e], axis=-1)

    pv_set      = set(PV_COLS)
    non_pv      = [c for c in sensor_cols if c not in pv_set]
    c2i         = {c: i for i, c in enumerate(non_pv)}
    ctrl_cv_idx = {ln: c2i[LOOPS[ln].cv] for ln in CTRL_LOOPS if LOOPS[ln].cv in c2i}

    print("Loading models...")
    plant, ctrls, TL = load_models(data)

    # ── Generate synthetic data (balanced: N_SYNTH per attack class) ──────────
    print(f"\nGenerating {N_SYNTH} synthetic windows per attack scenario...")
    np.random.seed(42)
    pv_real   = plant_data["pv_target_test"]
    real_trajs  = {sc: pv_real[sc_arr == sc] for sc in range(4)}
    synth_trajs = {}

    for sc_id in [1, 2, 3]:
        print(f"  {SC_LABELS[sc_id]}...", end=" ", flush=True)
        synth_trajs[sc_id] = generate_scenario(
            plant, ctrls, ctrl_cv_idx,
            plant_data, ctrl_data, sc_id, N_SYNTH, TL, NP)
        print(f"{len(synth_trajs[sc_id])} windows generated")

        # save CSV
        import pandas as pd
        rows = [{"window": w, "timestep": t,
                 "scenario_id": sc_id, "scenario": SC_LABELS[sc_id],
                 **{PV_SHORT[k]: float(synth_trajs[sc_id][w, t, k])
                    for k in range(NP)}}
                for w in range(len(synth_trajs[sc_id]))
                for t in range(TL)]
        save_path = OUT_DIR.parent / "data" / f"synthetic_{SC_LABELS[sc_id]}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"    CSV saved: {save_path.name}")

    # ── Extract features ──────────────────────────────────────────────────────
    print("\nExtracting statistical features...")
    X_real_feat = extract_features(pv_real)       # all real test windows
    y_real      = sc_arr

    X_synth_all = np.concatenate([synth_trajs[i] for i in [1,2,3]] +
                                  [real_trajs[0]])   # synthetic attacks + real normal
    # pure synthetic (all 4 classes synthetic)
    synth_normal = generate_scenario(plant, ctrls, ctrl_cv_idx,
                                      plant_data, ctrl_data, 0, N_SYNTH, TL, NP)
    print(f"  Normal synthetic: {len(synth_normal)} windows")

    X_pure_synth = np.concatenate([extract_features(synth_normal)] +
                                   [extract_features(synth_trajs[i]) for i in [1,2,3]])
    y_pure_synth = np.concatenate([np.zeros(N_SYNTH, dtype=int)] +
                                   [np.full(N_SYNTH, i, dtype=int) for i in [1,2,3]])

    X_mixed = np.concatenate([X_real_feat[sc_arr == 0],
                               extract_features(synth_trajs[1]),
                               extract_features(synth_trajs[2]),
                               extract_features(synth_trajs[3])])
    y_mixed = np.concatenate([np.zeros((sc_arr == 0).sum(), dtype=int),
                               np.ones(N_SYNTH, dtype=int),
                               np.full(N_SYNTH, 2, dtype=int),
                               np.full(N_SYNTH, 3, dtype=int)])

    # ── Plot 1: Data quality ──────────────────────────────────────────────────
    print("\n[Plot 1] Data quality (real vs synthetic trajectories)...")
    plot_data_quality(real_trajs, synth_trajs)

    # ── Experiment A: Baseline (imbalanced real only) ─────────────────────────
    print("\n[Experiment A] Baseline — train on imbalanced real data...")
    pred_A, _, f1_A, _ = run_clf(X_real_feat, y_real,
                                   X_real_feat, y_real,
                                   "Baseline: Real only (imbalanced)")

    # ── Experiment B: TRTS — train synthetic only, test real ─────────────────
    print("\n[Experiment B] TRTS — train on synthetic only, test on real...")
    pred_B, _, f1_B, (clf_trts, scaler_trts) = run_clf(X_pure_synth, y_pure_synth,
                                   X_real_feat,  y_real,
                                   "TRTS: Synthetic only → Real")

    # ── Experiment C: Mixed — real Normal + synthetic attacks ─────────────────
    print("\n[Experiment C] Mixed — real Normal + synthetic attacks, test on real...")
    pred_C, _, f1_C, _ = run_clf(X_mixed, y_mixed,
                                   X_real_feat, y_real,
                                   "Mixed: Real Normal + Synthetic attacks → Real")

    # ── Plot 2: Experiment comparison bar ─────────────────────────────────────
    print("\n[Plot 2] Experiment comparison...")
    exp_results = {
        "Baseline\n(Real only)":       {"macro_f1": f1_A},
        "TRTS\n(Synthetic only)":      {"macro_f1": f1_B},
        "Mixed\n(Real + Synthetic)":   {"macro_f1": f1_C},
    }
    plot_experiment_bar(exp_results)

    # ── Plot 3 & 4: Confusion matrices ────────────────────────────────────────
    print("[Plot 3] Confusion matrix — TRTS...")
    plot_confusion(y_real, pred_B,
                   "TRTS: Trained on Synthetic Only — Tested on Real",
                   "s3c_confusion_trts.png")

    print("[Plot 4] Confusion matrix — Mixed...")
    plot_confusion(y_real, pred_C,
                   "Mixed: Real Normal + Synthetic Attacks — Tested on Real",
                   "s3c_confusion_mixed.png")

    # ── Save stats ────────────────────────────────────────────────────────────
    stats = {
        "baseline_macro_f1": float(f1_A),
        "trts_macro_f1":     float(f1_B),
        "mixed_macro_f1":    float(f1_C),
        "n_synth_per_class": int(N_SYNTH),
        "real_class_counts": {SC_LABELS[i]: int((sc_arr==i).sum()) for i in range(4)},
    }
    with open(OUT_DIR / "s3c_classification_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: s3c_classification_stats.json")

    # ── Save TRTS classifier + scaler ─────────────────────────────────────────
    MODEL_DIR = ROOT / "outputs" / "classifiers"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_trts,    MODEL_DIR / "trts_rf_classifier.pkl")
    joblib.dump(scaler_trts, MODEL_DIR / "trts_rf_scaler.pkl")
    print(f"  TRTS classifier saved: {MODEL_DIR / 'trts_rf_classifier.pkl'}")
    print(f"  TRTS scaler    saved: {MODEL_DIR / 'trts_rf_scaler.pkl'}")

    print(f"\nAll Section 3 classification plots saved to: {OUT_DIR}/")
