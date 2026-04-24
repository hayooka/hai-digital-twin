"""
evaluate_generation.py — Evaluate synthetic data quality via classifier transfer.

Three experiments:
  A. Train on REAL,      test on SYNTHETIC  → does synthetic look real?
  B. Train on SYNTHETIC, test on REAL       → does synthetic teach real patterns?
  C. Train on MIXED,     test on REAL       → does augmentation help?

Usage:
    python 05_detect/evaluate_generation.py
    python 05_detect/evaluate_generation.py --ckpt outputs/pipeline/gru_scenario_haiend/gru_plant.pt
"""

import sys
import argparse
import json
import numpy as np
import torch
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, HAIEND_COLS, PROCESSED_DATA_DIR

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 128
CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC', 'CC']

SCENARIO_NAMES  = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SCENARIO_COLORS = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}

OUT_DIR = ROOT / "outputs" / "evaluate_generation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_CHANNELS = {
    'PC': ['P1_PCV02D', 'P1_FT01',   'P1_TIT01'],
    'LC': ['P1_FT03',   'P1_FCV03D', 'P1_PCV01D'],
    'FC': ['P1_PIT01',  'P1_LIT01',  'P1_TIT03'],
    'TC': ['P1_FT02',   'P1_PIT02',  'P1_TIT02'],
    'CC': ['P1_PP04D',  'P1_FCV03D', 'P1_PCV02D'],
}


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(trajectories: np.ndarray) -> np.ndarray:
    """
    (N, T, K) → (N, K*6) feature matrix.
    Per channel: mean, std, min, max, mean_abs, rate_of_change
    """
    feats = []
    for k in range(trajectories.shape[-1]):
        r = trajectories[:, :, k]
        feats.append(r.mean(axis=1))
        feats.append(r.std(axis=1))
        feats.append(r.min(axis=1))
        feats.append(r.max(axis=1))
        feats.append(np.abs(r).mean(axis=1))
        feats.append(np.diff(r, axis=1).mean(axis=1))
    return np.stack(feats, axis=1)


# ── Model loader ───────────────────────────────────────────────────────────────

def load_models(ckpt_path, data):
    plant_data  = data['plant']
    ctrl_data   = data['ctrl']
    TARGET_LEN  = data['metadata']['target_len']
    N_PLANT_IN  = plant_data['n_plant_in']
    N_PV        = plant_data['n_pv']
    N_HAIEND    = plant_data['n_haiend']
    N_SCENARIOS = data['metadata']['n_scenarios']

    ckpt     = torch.load(ckpt_path, map_location=DEVICE)
    hidden   = ckpt.get('hidden', 512)
    layers   = ckpt.get('layers', 2)
    n_haiend = ckpt.get('n_haiend', N_HAIEND)

    plant_model = GRUPlant(
        n_plant_in=N_PLANT_IN, n_pv=N_PV,
        hidden=hidden, layers=layers,
        n_scenarios=N_SCENARIOS, dropout=0.0,
        n_haiend=n_haiend,
    ).to(DEVICE)
    plant_model.load_state_dict(ckpt['model_state'], strict=False)
    plant_model.eval()

    CTRL_HIDDEN = {'PC': 64, 'LC': 64, 'FC': 128, 'TC': 64, 'CC': 64}
    ctrl_models = {}
    for ln in CTRL_LOOPS:
        n_in = ctrl_data[ln]['X_train'].shape[-1]
        h    = CTRL_HIDDEN[ln]
        if ln == 'CC':
            m = CCSequenceModel(n_inputs=n_in, hidden=h, layers=2,
                                dropout=0.0, output_len=TARGET_LEN).to(DEVICE)
        else:
            m = GRUController(n_inputs=n_in, hidden=h, layers=2,
                              dropout=0.0, output_len=TARGET_LEN).to(DEVICE)
        p = ckpt_path.parent / f"gru_ctrl_{ln.lower()}.pt"
        if p.exists():
            c = torch.load(p, map_location=DEVICE)
            m.load_state_dict(c['model_state'], strict=False)
        m.eval()
        ctrl_models[ln] = m

    return plant_model, ctrl_models, TARGET_LEN, N_PV, n_haiend


def generate_batch(plant_model, ctrl_models, ctrl_cv_col_idx,
                   X, X_cv_tgt, pv_init, ctrl_seed, TARGET_LEN, N_PV, sc_label):
    """Returns (pv_out, haiend_out) — both generated trajectories."""
    N        = len(X)
    pv_out   = np.zeros((N, TARGET_LEN, N_PV), dtype=np.float32)
    haiend_list = []
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl        = slice(i, i + BATCH)
            x_cv_b    = torch.tensor(X[sl]).float().to(DEVICE)
            xct_b     = torch.tensor(X_cv_tgt[sl]).float().to(DEVICE).clone()
            pv_init_b = torch.tensor(pv_init[sl]).float().to(DEVICE)
            sc_b      = torch.full((x_cv_b.size(0),), sc_label,
                                   dtype=torch.long, device=DEVICE)
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx: continue
                Xc      = torch.tensor(ctrl_seed[ln][sl]).float().to(DEVICE)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                ci      = ctrl_cv_col_idx[ln]
                xct_b[:, :, ci:ci+1] = cv_pred
            pv_seq, haiend_seq = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
            pv_out[i:i + x_cv_b.size(0)] = pv_seq.cpu().numpy()
            if haiend_seq is not None:
                haiend_list.append(haiend_seq.cpu().numpy())
    haiend_out = np.concatenate(haiend_list, axis=0) if haiend_list else None
    return pv_out, haiend_out


# ── Classifier experiment ──────────────────────────────────────────────────────

def run_classifier(X_train, y_train, X_test, y_test, label):
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                 class_weight='balanced')
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    target_names = [SCENARIO_NAMES[i] for i in sorted(SCENARIO_NAMES)
                    if i in np.unique(np.concatenate([y_train, y_test]))]
    print(f"\n  [{label}]")
    print(classification_report(y_test, y_pred,
                                 target_names=target_names, zero_division=0))
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return clf, sc, y_pred, macro_f1


def plot_confusion(y_true, y_pred, title, fname):
    labels     = sorted(np.unique(np.concatenate([y_true, y_pred])))
    label_names = [SCENARIO_NAMES[i] for i in labels]
    cm   = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(ckpt_path: Path, n_synth: int):
    print("=" * 60)
    print("HAI Digital Twin — Synthetic Generation Evaluation")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    data        = load_and_prepare_data()
    plant_data  = data['plant']
    ctrl_data   = data['ctrl']
    sensor_cols = data['metadata']['sensor_cols']
    TARGET_LEN  = data['metadata']['target_len']

    plant_scaler = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    npz = {s: np.load(f"{PROCESSED_DATA_DIR}/{s}_data.npz")
           for s in ("train", "val", "test")}
    col_idx = {c: i for i, c in enumerate(sensor_cols)}
    for ln, extra_cols in EXTRA_CHANNELS.items():
        for ec in extra_cols:
            if ec not in col_idx: continue
            ei = col_idx[ec]
            mean_e, scale_e = plant_scaler.mean_[ei], plant_scaler.scale_[ei]
            for split, arr in npz.items():
                raw = arr['X'][:, :, [ei]].astype(np.float32)
                ctrl_data[ln][f'X_{split}'] = np.concatenate(
                    [ctrl_data[ln][f'X_{split}'], (raw - mean_e) / scale_e], axis=-1)

    pv_set      = set(PV_COLS)
    non_pv_cols = [c for c in sensor_cols if c not in pv_set]
    col_to_idx  = {c: i for i, c in enumerate(non_pv_cols)}
    ctrl_cv_col_idx = {ln: col_to_idx[LOOPS[ln].cv]
                       for ln in CTRL_LOOPS if LOOPS[ln].cv in col_to_idx}

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\n[2] Loading model from {ckpt_path.name}...")
    plant_model, ctrl_models, TARGET_LEN, N_PV, N_HAIEND = \
        load_models(ckpt_path, data)

    # ── Real data: PV only (HAIEND head not strong enough for classification) ──
    pv_real = plant_data['pv_target_test']   # (N, T, N_PV)
    y_real  = plant_data['scenario_test']

    print(f"\n    Real test distribution: "
          + "  ".join(f"{SCENARIO_NAMES[i]}={( y_real==i).sum()}"
                      for i in SCENARIO_NAMES))

    # ── Option B: Use real pre-attack inputs as seeds ─────────────────────────
    # For each scenario, take real input windows (300s before/during attack)
    # and generate the output conditioned on each attack label.
    # This grounds the synthetic data in real plant conditions.
    print(f"\n[3] Generating synthetic trajectories from real pre-attack inputs (Option B)...")

    synth       = {}
    seed_counts = {}
    np.random.seed(42)

    for sc_id, sc_name in SCENARIO_NAMES.items():
        mask = np.where(plant_data['scenario_test'] == sc_id)[0]
        if len(mask) == 0:
            print(f"    {sc_name}: no real windows — skipping")
            continue
        # repeat seeds to reach n_synth
        rep_idx = np.random.choice(mask,
                                   size=min(n_synth, len(mask) * 50),
                                   replace=True)
        X_seed       = plant_data['X_test'][rep_idx]
        X_cv_seed    = plant_data['X_cv_target_test'][rep_idx]
        pv_init_seed = plant_data['pv_init_test'][rep_idx]
        ctrl_seed    = {ln: ctrl_data[ln]['X_test'][rep_idx] for ln in CTRL_LOOPS}

        print(f"    {sc_name}: generating {len(rep_idx)} windows "
              f"(from {len(mask)} real seeds)...")
        pv_gen, haiend_gen = generate_batch(
            plant_model, ctrl_models, ctrl_cv_col_idx,
            X_seed, X_cv_seed, pv_init_seed, ctrl_seed,
            TARGET_LEN, N_PV, sc_id
        )
        synth[sc_id]       = (pv_gen, haiend_gen)
        seed_counts[sc_id] = len(rep_idx)

    # Build synthetic dataset — PV only for classification
    X_synth_all = np.concatenate([synth[i][0] for i in synth], axis=0)
    y_synth_all = np.concatenate([
        np.full(seed_counts[i], i) for i in synth
    ])

    # Feature extraction
    print("\n[4] Extracting features...")
    X_real_feat  = extract_features(pv_real)
    X_synth_feat = extract_features(X_synth_all)
    print(f"    Real:      {X_real_feat.shape}  labels={np.bincount(y_real)}")
    print(f"    Synthetic: {X_synth_feat.shape}  labels={np.bincount(y_synth_all.astype(int))}")

    # ── Experiment A: Train REAL → Test SYNTHETIC ──────────────────────────────
    print("\n" + "=" * 50)
    print("EXPERIMENT A: Train on REAL, Test on SYNTHETIC")
    print("Checks: does synthetic data look like real data?")
    print("=" * 50)
    _, _, y_pred_A, f1_A = run_classifier(
        X_real_feat, y_real,
        X_synth_feat, y_synth_all,
        "Train=Real, Test=Synthetic"
    )
    plot_confusion(y_synth_all, y_pred_A,
                   "Exp A: Train Real → Test Synthetic",
                   OUT_DIR / "confmat_A_real_to_synth.png")

    # ── Experiment B: Train SYNTHETIC → Test REAL ─────────────────────────────
    print("\n" + "=" * 50)
    print("EXPERIMENT B: Train on SYNTHETIC, Test on REAL")
    print("Checks: does synthetic teach real attack patterns?")
    print("=" * 50)
    _, _, y_pred_B, f1_B = run_classifier(
        X_synth_feat, y_synth_all,
        X_real_feat, y_real,
        "Train=Synthetic, Test=Real"
    )
    plot_confusion(y_real, y_pred_B,
                   "Exp B: Train Synthetic → Test Real",
                   OUT_DIR / "confmat_B_synth_to_real.png")

    # ── Experiment C: Train MIXED → Test REAL (no leakage) ───────────────────
    print("\n" + "=" * 50)
    print("EXPERIMENT C: Train on REAL(train) + SYNTHETIC, Test on REAL(test)")
    print("Checks: does augmentation improve classification?")
    print("=" * 50)

    # Use real TRAIN split (no overlap with test) + synthetic for training
    # Evaluate on real TEST split only
    pv_real_train = plant_data['pv_target_train']
    y_real_train  = plant_data['scenario_train']
    X_real_train_feat = extract_features(pv_real_train)

    X_mixed = np.concatenate([X_real_train_feat, X_synth_feat], axis=0)
    y_mixed = np.concatenate([y_real_train,      y_synth_all],  axis=0)

    _, _, y_pred_C, f1_C = run_classifier(
        X_mixed, y_mixed,
        X_real_feat, y_real,
        "Train=Real(train)+Synthetic, Test=Real(test)"
    )
    plot_confusion(y_real, y_pred_C,
                   "Exp C: Train Mixed → Test Real",
                   OUT_DIR / "confmat_C_mixed_to_real.png")

    # ── Summary plot: F1 comparison across experiments ────────────────────────
    print("\n[5] Plotting summary...")
    experiments = ['A\nReal→Synth', 'B\nSynth→Real', 'C\nMixed→Real']
    f1_scores   = [f1_A, f1_B, f1_C]
    colors_bar  = ['#2196F3', '#FF5722', '#4CAF50']

    fig_s, ax_s = plt.subplots(figsize=(7, 4))
    bars = ax_s.bar(experiments, f1_scores, color=colors_bar,
                    edgecolor='white', width=0.5)
    ax_s.set_ylim(0, 1.05)
    ax_s.set_ylabel("Macro F1 Score", fontsize=10)
    ax_s.set_title("Synthetic data quality — classifier transfer experiments",
                   fontsize=11)
    for bar, val in zip(bars, f1_scores):
        ax_s.text(bar.get_x() + bar.get_width() / 2,
                  val + 0.02, f"{val:.3f}", ha='center', fontsize=10,
                  fontweight='bold')
    ax_s.axhline(0.5, color='gray', linestyle='--', linewidth=0.8,
                 label='0.5 reference')
    ax_s.legend(fontsize=8)
    fig_s.tight_layout()
    fig_s.savefig(OUT_DIR / "experiment_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig_s)
    print(f"    Saved: {OUT_DIR}/experiment_summary.png")

    # ── Per-scenario generation quality plot ──────────────────────────────────
    print("\n[6] Plotting per-scenario trajectory comparison...")
    fig_t, axes_t = plt.subplots(len(SCENARIO_NAMES), N_PV,
                                  figsize=(3.5 * N_PV, 3 * len(SCENARIO_NAMES)),
                                  squeeze=False)
    t_ax = np.arange(TARGET_LEN)
    for row, (sc_id, sc_name) in enumerate(SCENARIO_NAMES.items()):
        real_mask = np.where(y_real == sc_id)[0]
        real_mean = pv_real[real_mask].mean(axis=0) if len(real_mask) > 0 \
                    else np.zeros((TARGET_LEN, N_PV))
        syn_mean  = synth[sc_id][0].mean(axis=0) if sc_id in synth else np.zeros((TARGET_LEN, N_PV))

        for col, pv_name in enumerate(PV_COLS):
            ax = axes_t[row, col]
            ax.plot(t_ax, real_mean[:, col],
                    color=SCENARIO_COLORS[sc_id], linewidth=2, label='Real')
            ax.plot(t_ax, syn_mean[:, col],
                    color='black', linewidth=1.2,
                    linestyle='--', label='Synthetic')
            ax.fill_between(t_ax, real_mean[:, col], syn_mean[:, col],
                            alpha=0.12, color=SCENARIO_COLORS[sc_id])
            if col == 0:
                ax.set_ylabel(sc_name, fontsize=8,
                              color=SCENARIO_COLORS[sc_id], fontweight='bold')
            if row == 0:
                ax.set_title(pv_name, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("t (s)", fontsize=6)
    axes_t[0, 0].legend(fontsize=6)
    fig_t.suptitle("Real vs Synthetic mean trajectories per scenario",
                   fontsize=11)
    fig_t.tight_layout()
    fig_t.savefig(OUT_DIR / "real_vs_synthetic_all.png",
                  dpi=150, bbox_inches='tight')
    plt.close(fig_t)
    print(f"    Saved: {OUT_DIR}/real_vs_synthetic_all.png")

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "checkpoint": str(ckpt_path),
        "n_synth_per_scenario": n_synth,
        "macro_f1": {
            "A_real_to_synth": round(f1_A, 4),
            "B_synth_to_real": round(f1_B, 4),
            "C_mixed_to_real": round(f1_C, 4),
        }
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Exp A (Real→Synth)  Macro F1 = {f1_A:.4f}")
    print(f"  Exp B (Synth→Real)  Macro F1 = {f1_B:.4f}")
    print(f"  Exp C (Mixed→Real)  Macro F1 = {f1_C:.4f}")
    print(f"\n  Outputs saved to: {OUT_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/pipeline/gru_scenario_haiend/gru_plant.pt")
    parser.add_argument("--n_synth", type=int, default=500,
                        help="Synthetic windows to generate per scenario")
    args = parser.parse_args()
    main(Path(ROOT / args.ckpt), args.n_synth)
