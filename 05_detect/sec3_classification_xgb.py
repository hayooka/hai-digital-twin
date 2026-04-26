"""
sec3_classification_xgb.py — Section 3 Classification using XGBoost

Identical methodology to sec3_classification.py (RandomForest) but uses
XGBoost with inverse-frequency class weighting instead.

Experiments:
  A. Baseline : XGBoost on real data (80% train / 20% test, stratified)
  B. TRTS     : train on synthetic only (train-seeded), test on real test
  C. Mixed    : train on real Normal (train) + synthetic attacks, test on real test

Results saved to:
  report_plots/figures/s3_experement/s3c_classification_stats.json  (overwrite)
  report_plots/figures/s3c_experiment_bar.png                        (overwrite)
  report_plots/figures/s3c_confusion_{A,B,C}.png                     (overwrite)
"""

import sys, json
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 128
CKPT_DIR   = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"
OUT_DIR    = ROOT / "report_plots" / "figures"
STATS_DIR  = ROOT / "report_plots" / "figures" / "s3_experement"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

CTRL_LOOPS  = ["PC", "LC", "FC", "TC", "CC"]
CTRL_H      = {"PC": 64, "LC": 64, "FC": 128, "TC": 64, "CC": 64}
N_SYNTH     = 3292
SC_LABELS   = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SC_COLORS   = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}
PV_SHORT    = [c.replace("P1_", "") for c in PV_COLS]

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
                      sc_id, n_target, TL, NP, split="train"):
    sc_arr = plant_data[f"scenario_{split}"]
    mask   = np.where(sc_arr == sc_id)[0]
    if len(mask) == 0:
        print(f"    WARNING: sc_id={sc_id} not found in {split}, using all train windows")
        mask = np.arange(len(sc_arr))
    rep = np.random.choice(mask, size=n_target, replace=True)
    out = np.zeros((n_target, TL, NP), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_target, BATCH):
            sl  = slice(i, i + BATCH)
            idx = rep[sl]
            xb  = torch.tensor(plant_data[f"X_{split}"][idx]).float().to(DEVICE)
            xcb = torch.tensor(plant_data[f"X_cv_target_{split}"][idx]).float().to(DEVICE).clone()
            pb  = torch.tensor(plant_data[f"pv_init_{split}"][idx]).float().to(DEVICE)
            sb  = torch.full((xb.size(0),), sc_id, dtype=torch.long, device=DEVICE)
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_idx: continue
                cp = ctrls[ln].predict(
                    torch.tensor(ctrl_data[ln][f"X_{split}"][idx]).float().to(DEVICE),
                    target_len=TL)
                xcb[:, :, ctrl_cv_idx[ln]:ctrl_cv_idx[ln]+1] = cp
            pv, _ = plant.predict(xb, xcb, pb, sb)
            out[i:i + xb.size(0)] = pv.cpu().numpy()
    return out


def extract_features(traj):
    """(N, T, K) -> (N, K*6) statistical features."""
    feats = []
    for k in range(traj.shape[-1]):
        r = traj[:, :, k]
        feats += [r.mean(1), r.std(1), r.min(1), r.max(1),
                  np.abs(r).mean(1), np.diff(r, axis=1).mean(1)]
    return np.stack(feats, axis=1)


def make_sample_weights(y):
    """Inverse-frequency sample weights for imbalanced classes."""
    counts = Counter(y)
    total  = len(y)
    return np.array([total / (len(counts) * counts[yi]) for yi in y], dtype=np.float32)


def run_clf(X_tr, y_tr, X_te, y_te, label):
    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    sw = make_sample_weights(y_tr)

    clf = XGBClassifier(
        n_estimators  = 500,
        max_depth      = 6,
        learning_rate  = 0.05,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder = False,
        eval_metric    = "mlogloss",
        random_state   = 42,
        n_jobs         = -1,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    y_pred    = clf.predict(X_te)
    macro     = f1_score(y_te, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_te, y_pred, average=None, zero_division=0, labels=[0,1,2,3])
    print(f"\n  [{label}]  Macro F1 = {macro:.4f}")
    print(classification_report(y_te, y_pred,
          target_names=[SC_LABELS[i] for i in range(4)], zero_division=0))
    return y_pred, macro, per_class, (clf, sc)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_experiment_bar(results):
    names  = list(results.keys())
    macros = [results[n]["macro_f1"] for n in names]
    colors = ["#90A4AE", "#FF5722", "#4CAF50"]
    descs  = [
        "Real data\n(80/20 split)",
        "Synthetic only\n(TRTS, train-seeded)",
        "Real Normal +\nSynthetic attacks",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, macros, color=colors, alpha=0.88, width=0.5)
    for bar, v in zip(bars, macros):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    for bar, desc in zip(bars, descs):
        ax.text(bar.get_x() + bar.get_width()/2, -0.10, desc,
                ha="center", va="top", fontsize=8, color="gray",
                transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro F1 Score (4-class)", fontsize=12)
    ax.set_title("Attack Classification (XGBoost) — Correct Train/Test Split, Train-Seeded Synthetic",
                 fontsize=13, fontweight="bold")
    ax.axhline(0.25, color="gray", linestyle=":", lw=1.2, label="Random baseline (4-class)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = OUT_DIR / "s3c_experiment_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_confusion(y_true, y_pred, title, fname):
    labels  = [SC_LABELS[i] for i in range(4)]
    cm      = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, subtitle, fmt in zip(
        axes, [cm, cm_norm], ["Counts", "Normalised"], [".0f", ".2f"]
    ):
        im = ax.imshow(data, cmap="Blues", interpolation="nearest")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
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
    fig.suptitle(f"{title}  —  Macro F1 = {macro:.3f}", fontsize=13, fontweight="bold")
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

    pv_test = plant_data["pv_target_test"]
    sc_test = plant_data["scenario_test"]

    # Generate synthetic data seeded ONLY from train windows
    print(f"\nGenerating {N_SYNTH} synthetic windows per scenario (train seeds)...")
    np.random.seed(42)
    synth_trajs = {}
    for sc_id in [1, 2, 3]:
        print(f"  Generating {SC_LABELS[sc_id]}...", end=" ", flush=True)
        synth_trajs[sc_id] = generate_scenario(
            plant, ctrls, ctrl_cv_idx,
            plant_data, ctrl_data, sc_id, N_SYNTH, TL, NP, split="train")
        print(f"{len(synth_trajs[sc_id])} windows")

    print(f"  Generating Normal (synthetic)...", end=" ", flush=True)
    synth_normal = generate_scenario(
        plant, ctrls, ctrl_cv_idx,
        plant_data, ctrl_data, 0, N_SYNTH, TL, NP, split="train")
    print(f"{len(synth_normal)} windows")

    # Feature extraction
    print("\nExtracting features...")
    pv_train = plant_data["pv_target_train"]
    sc_train = plant_data["scenario_train"]

    X_real_train_feat = extract_features(pv_train)
    y_real_train      = sc_train
    X_real_test_feat  = extract_features(pv_test)
    y_real_test       = sc_test

    X_synth_atk  = np.concatenate([extract_features(synth_trajs[i]) for i in [1,2,3]])
    y_synth_atk  = np.concatenate([np.full(N_SYNTH, i, dtype=int) for i in [1,2,3]])
    X_synth_norm = extract_features(synth_normal)
    y_synth_norm = np.zeros(N_SYNTH, dtype=int)

    X_pure_synth = np.concatenate([X_synth_norm, X_synth_atk])
    y_pure_synth = np.concatenate([y_synth_norm, y_synth_atk])

    real_norm_mask = sc_train == 0
    X_mixed = np.concatenate([X_real_train_feat[real_norm_mask], X_synth_atk])
    y_mixed = np.concatenate([y_real_train[real_norm_mask], y_synth_atk])

    # Experiment A: real data, proper 80/20 stratified split
    print("\n[Experiment A] Baseline — XGBoost on real data, stratified 80/20 split...")
    X_A_tr, X_A_te, y_A_tr, y_A_te = train_test_split(
        X_real_train_feat, y_real_train,
        test_size=0.20, random_state=42, stratify=y_real_train)
    pred_A, f1_A, pc_A, _ = run_clf(X_A_tr, y_A_tr, X_A_te, y_A_te,
                                     "Baseline: Real 80/20")

    # Experiment B: TRTS — synthetic only, test on real held-out test
    print("\n[Experiment B] TRTS — train-seeded synthetic only, test on real test...")
    pred_B, f1_B, pc_B, (clf_trts, scaler_trts) = run_clf(
        X_pure_synth, y_pure_synth, X_real_test_feat, y_real_test,
        "TRTS: Synthetic → Real test")

    # Experiment C: mixed — real Normal + synthetic attacks, test on real test
    print("\n[Experiment C] Mixed — real Normal + synthetic attacks, test on real test...")
    pred_C, f1_C, pc_C, (clf_mixed, scaler_mixed) = run_clf(
        X_mixed, y_mixed, X_real_test_feat, y_real_test,
        "Mixed: Real Normal + Synthetic attacks → Real test")

    # Plots
    exp_results = {
        "Baseline\n(Real 80/20)":       {"macro_f1": f1_A, "per_class_f1": pc_A},
        "TRTS\n(Synth→Real)":           {"macro_f1": f1_B, "per_class_f1": pc_B},
        "Mixed\n(Real Norm+Synth Atk)": {"macro_f1": f1_C, "per_class_f1": pc_C},
    }

    print("\n[Plot] Experiment comparison bar...")
    plot_experiment_bar(exp_results)

    print("[Plot] Confusion matrix — Baseline (A)...")
    plot_confusion(y_A_te, pred_A,
                   "XGBoost Baseline: Real data 80/20 split", "s3c_confusion_A.png")

    print("[Plot] Confusion matrix — TRTS (B)...")
    plot_confusion(y_real_test, pred_B,
                   "XGBoost TRTS: Train-seeded Synthetic → Real test", "s3c_confusion_B.png")

    print("[Plot] Confusion matrix — Mixed (C)...")
    plot_confusion(y_real_test, pred_C,
                   "XGBoost Mixed: Real Normal + Synthetic attacks → Real test", "s3c_confusion_C.png")

    # Save stats — overwrites the flawed s3_experement stats with correct values
    stats = {
        "model": "XGBClassifier",
        "features": "30 (6 stats x 5 PVs, 180s window)",
        "baseline_macro_f1": float(f1_A),
        "trts_macro_f1":     float(f1_B),
        "mixed_macro_f1":    float(f1_C),
        "per_class_f1": {
            "Baseline": {SC_LABELS[i]: float(pc_A[i]) for i in range(4)},
            "TRTS":     {SC_LABELS[i]: float(pc_B[i]) for i in range(4)},
            "Mixed":    {SC_LABELS[i]: float(pc_C[i]) for i in range(4)},
        },
        "n_synth_per_class": int(N_SYNTH),
        "real_class_counts": {SC_LABELS[i]: int((sc_test == i).sum()) for i in range(4)},
    }
    stats_path = STATS_DIR / "s3c_classification_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    # Save classifiers
    MODEL_DIR = ROOT / "outputs" / "classifiers"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_trts,    MODEL_DIR / "trts_xgb_classifier.pkl")
    joblib.dump(scaler_trts, MODEL_DIR / "trts_xgb_scaler.pkl")
    joblib.dump(clf_mixed,   MODEL_DIR / "mixed_xgb_classifier.pkl")
    joblib.dump(scaler_mixed, MODEL_DIR / "mixed_xgb_scaler.pkl")
    print(f"  Classifiers saved to: {MODEL_DIR}/")

    print("\n=== RESULTS SUMMARY ===")
    for name, r in exp_results.items():
        label = name.replace("\n", " ")
        pc    = r["per_class_f1"]
        print(f"  {label:<40s} macro={r['macro_f1']:.3f}  "
              f"Normal={pc[0]:.2f}  AP_no={pc[1]:.2f}  "
              f"AP_with={pc[2]:.2f}  AE_no={pc[3]:.2f}")
