"""
sec2_generation.py — Section 2: Synthetic Data Generation

Produces:
  figures/s2_gen_Normal.png       — synthetic vs real mean trajectories (5 PVs stacked)
  figures/s2_gen_AP_no.png
  figures/s2_gen_AP_with.png
  figures/s2_gen_AE_no.png
  figures/s2_classifier_results.png — TSTR/TRTS/Mixed F1 bar chart

Model used: gru_scenario_weighted

Usage:
    python report_plots/code/sec2_generation.py
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
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 128
CTRL_LOOPS = ["PC", "LC", "FC", "TC", "CC"]
CKPT_DIR   = ROOT / "outputs" / "pipeline" / "gru_scenario_weighted"
OUT_DIR    = ROOT / "report_plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SYNTH = 200  # synthetic windows per scenario

PV_SHORT = [p.replace("P1_", "") for p in PV_COLS]
PV_LOOPS = ["PC", "LC", "FC", "TC", "CC"]

SCENARIOS = {
    0: ("Normal",  "#2196F3"),
    1: ("AP_no",   "#FF5722"),
    2: ("AP_with", "#E91E63"),
    3: ("AE_no",   "#9C27B0"),
}

EXTRA_CHANNELS = {
    "PC": ["P1_PCV02D", "P1_FT01",   "P1_TIT01"],
    "LC": ["P1_FT03",   "P1_FCV03D", "P1_PCV01D"],
    "FC": ["P1_PIT01",  "P1_LIT01",  "P1_TIT03"],
    "TC": ["P1_FT02",   "P1_PIT02",  "P1_TIT02"],
    "CC": ["P1_PP04D",  "P1_FCV03D", "P1_PCV02D"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_models(data):
    pd   = data["plant"]
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

    CTRL_H = {"PC": 64, "LC": 64, "FC": 128, "TC": 64, "CC": 64}
    ctrls  = {}
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


def generate_batch(plant, ctrls, ctrl_cv_idx, X, Xcv, pvi, ctrl_seed, TL, NP, sc_label):
    N   = len(X)
    out = np.zeros((N, TL, NP), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl  = slice(i, i + BATCH)
            xb  = torch.tensor(X[sl]).float().to(DEVICE)
            xcb = torch.tensor(Xcv[sl]).float().to(DEVICE).clone()
            pb  = torch.tensor(pvi[sl]).float().to(DEVICE)
            sb  = torch.full((xb.size(0),), sc_label, dtype=torch.long, device=DEVICE)
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_idx: continue
                cp = ctrls[ln].predict(
                    torch.tensor(ctrl_seed[ln][sl]).float().to(DEVICE), target_len=TL)
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


# ── Plot: generation overlay (5 PVs stacked) ─────────────────────────────────

def plot_generation(sc_name, color, synth_traj, real_traj, TL, n_samples=10):
    t          = np.arange(TL)
    real_mean  = real_traj.mean(axis=0)
    synth_mean = synth_traj.mean(axis=0)
    np.random.seed(42)
    sample_idx = np.random.choice(len(synth_traj),
                                   size=min(n_samples, len(synth_traj)),
                                   replace=False)

    fig, axes = plt.subplots(len(PV_COLS), 1,
                              figsize=(10, 2.8 * len(PV_COLS)),
                              sharex=True)

    for col, (ax, pv, loop) in enumerate(zip(axes, PV_SHORT, PV_LOOPS)):
        # individual synthetic samples
        for i, si in enumerate(sample_idx):
            ax.plot(t, synth_traj[si, :, col], color=color, lw=0.7,
                    alpha=0.2, label="Synthetic samples" if i == 0 else "")
        # means
        ax.plot(t, synth_mean[:, col], color=color, lw=2.0, label="Synthetic mean")
        ax.plot(t, real_mean[:, col],  color="black", lw=1.8,
                linestyle="--", label="Real mean")

        ax.set_ylabel(f"{pv} [{loop}]", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)

    handles = [
        plt.Line2D([0], [0], color=color,   lw=0.7, alpha=0.5, label="Synthetic samples"),
        plt.Line2D([0], [0], color=color,   lw=2.0,             label="Synthetic mean"),
        plt.Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="Real mean"),
    ]
    axes[0].legend(handles=handles, fontsize=9, ncol=3, loc="upper left")

    fig.suptitle(f"Scenario: {sc_name} — Synthetic Generation vs Real Mean",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = OUT_DIR / f"s2_gen_{sc_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot: classifier transfer results ────────────────────────────────────────

def plot_classifier_results(results):
    experiments = list(results.keys())
    f1_vals     = [results[e]["macro_f1"] for e in experiments]
    colors      = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(experiments, f1_vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Synthetic Data Quality — Classifier Transfer Experiments",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=results.get("TRTS", {}).get("macro_f1", 0),
               color="gray", linestyle=":", lw=1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Experiment descriptions
    descs = {
        "TSTR\n(Real→Synth)":    "Does synthetic\nlook real?",
        "TRTS\n(Synth→Real)":    "Does synthetic\nteach real?",
        "Mixed→Real":            "Does augmentation\nhelp?",
    }
    for bar, exp in zip(bars, experiments):
        desc = descs.get(exp, "")
        ax.text(bar.get_x() + bar.get_width()/2,
                -0.12, desc, ha="center", va="top",
                fontsize=8, color="gray",
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    path = OUT_DIR / "s2_classifier_results.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data       = load_and_prepare_data()
    plant_data = data["plant"]
    ctrl_data  = data["ctrl"]
    sensor_cols = data["metadata"]["sensor_cols"]
    TL          = data["metadata"]["target_len"]
    NP          = plant_data["n_pv"]

    # augment controller data with causal channels
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

    # ── Generate synthetic data per scenario ──────────────────────────────────
    print("\nGenerating synthetic trajectories...")
    pv_real = plant_data["pv_target_test"]
    sc_arr  = plant_data["scenario_test"]
    synth   = {}
    np.random.seed(42)

    for sc_id, (sc_name, color) in SCENARIOS.items():
        mask = np.where(sc_arr == sc_id)[0]
        if len(mask) == 0:
            print(f"  {sc_name}: no test windows — skipping")
            continue
        rep_idx = np.random.choice(mask, size=min(N_SYNTH, len(mask)*50), replace=True)
        print(f"  {sc_name}: generating {len(rep_idx)} windows...")
        pv_gen = generate_batch(
            plant, ctrls, ctrl_cv_idx,
            plant_data["X_test"][rep_idx],
            plant_data["X_cv_target_test"][rep_idx],
            plant_data["pv_init_test"][rep_idx],
            {ln: ctrl_data[ln]["X_test"][rep_idx] for ln in CTRL_LOOPS},
            TL, NP, sc_id
        )
        synth[sc_id] = pv_gen

        # generation plot
        real_sc = pv_real[mask]
        plot_generation(sc_name, color, pv_gen, real_sc, TL)

    # ── Classifier transfer experiments ───────────────────────────────────────
    print("\nRunning classifier transfer experiments...")

    X_synth_all = np.concatenate([synth[i] for i in synth], axis=0)
    y_synth_all = np.concatenate([np.full(len(synth[i]), i) for i in synth])
    X_real_feat  = extract_features(pv_real)
    X_synth_feat = extract_features(X_synth_all)
    y_real       = sc_arr

    clf_results = {}

    def run_clf(Xtr, ytr, Xte, yte, name):
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                     class_weight="balanced")
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        f1 = f1_score(yte, yp, average="macro", zero_division=0)
        print(f"\n  [{name}]  Macro F1 = {f1:.4f}")
        print(classification_report(yte, yp,
              target_names=[SCENARIOS[i][0] for i in sorted(SCENARIOS)],
              zero_division=0))
        return f1

    clf_results["TSTR\n(Real→Synth)"] = {
        "macro_f1": run_clf(X_real_feat, y_real, X_synth_feat, y_synth_all,
                            "A: Train REAL, Test SYNTHETIC")
    }
    clf_results["TRTS\n(Synth→Real)"] = {
        "macro_f1": run_clf(X_synth_feat, y_synth_all, X_real_feat, y_real,
                            "B: Train SYNTHETIC, Test REAL")
    }
    # Mixed: 50% real + 50% synthetic for training
    n_mix = min(len(X_real_feat), len(X_synth_feat))
    r_idx = np.random.choice(len(X_real_feat),  n_mix, replace=False)
    s_idx = np.random.choice(len(X_synth_feat), n_mix, replace=False)
    X_mix = np.concatenate([X_real_feat[r_idx], X_synth_feat[s_idx]])
    y_mix = np.concatenate([y_real[r_idx],      y_synth_all[s_idx]])
    clf_results["Mixed→Real"] = {
        "macro_f1": run_clf(X_mix, y_mix, X_real_feat, y_real,
                            "C: Train MIXED, Test REAL")
    }

    plot_classifier_results(clf_results)
    print(f"\nAll Section 2 plots saved to: {OUT_DIR}/")
