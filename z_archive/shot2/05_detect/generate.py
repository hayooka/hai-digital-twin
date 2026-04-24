"""
generate.py — Synthetic attack data generation using the GRU digital twin.

Takes normal operating windows and generates all 4 scenario variants:
  - Label 0: Normal
  - Label 1: AP_no  (Actuator Pollution, no combustion)
  - Label 2: AP_with (Actuator Pollution, with combustion)
  - Label 3: AE_no  (Actuator Emulation, no combustion)

This solves the class imbalance problem in the HAI dataset:
  Normal:   10,148 windows  →  already plenty
  AP_no:        98 windows  →  generate thousands
  AP_with:      20 windows  →  generate thousands
  AE_no:        16 windows  →  generate thousands

Usage:
    python 05_detect/generate.py
    python 05_detect/generate.py --ckpt outputs/pipeline/gru_scenario_haiend/gru_plant.pt
    python 05_detect/generate.py --n_samples 500
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

OUT_DIR = ROOT / "outputs" / "generate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_CHANNELS = {
    'PC': ['P1_PCV02D', 'P1_FT01',   'P1_TIT01'],
    'LC': ['P1_FT03',   'P1_FCV03D', 'P1_PCV01D'],
    'FC': ['P1_PIT01',  'P1_LIT01',  'P1_TIT03'],
    'TC': ['P1_FT02',   'P1_PIT02',  'P1_TIT02'],
    'CC': ['P1_PP04D',  'P1_FCV03D', 'P1_PCV02D'],
}


# ── Load models ────────────────────────────────────────────────────────────────

def load_models(ckpt_path, data):
    plant_data  = data['plant']
    ctrl_data   = data['ctrl']
    TARGET_LEN  = data['metadata']['target_len']
    N_PLANT_IN  = plant_data['n_plant_in']
    N_PV        = plant_data['n_pv']
    N_HAIEND    = plant_data['n_haiend']
    N_SCENARIOS = data['metadata']['n_scenarios']

    ckpt    = torch.load(ckpt_path, map_location=DEVICE)
    hidden  = ckpt.get('hidden', 512)
    layers  = ckpt.get('layers', 2)
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
    ckpt_dir = ckpt_path.parent
    for ln in CTRL_LOOPS:
        n_in = ctrl_data[ln]['X_train'].shape[-1]
        h    = CTRL_HIDDEN[ln]
        if ln == 'CC':
            m = CCSequenceModel(n_inputs=n_in, hidden=h, layers=2,
                                dropout=0.0, output_len=TARGET_LEN).to(DEVICE)
        else:
            m = GRUController(n_inputs=n_in, hidden=h, layers=2,
                              dropout=0.0, output_len=TARGET_LEN).to(DEVICE)
        p = ckpt_dir / f"gru_ctrl_{ln.lower()}.pt"
        if p.exists():
            c = torch.load(p, map_location=DEVICE)
            m.load_state_dict(c['model_state'], strict=False)
        m.eval()
        ctrl_models[ln] = m

    return plant_model, ctrl_models, TARGET_LEN, N_PV, n_haiend


# ── Core: generate trajectories for a given scenario label ────────────────────

def generate_for_scenario(plant_model, ctrl_models, ctrl_cv_col_idx,
                           X, X_cv_tgt, pv_init, ctrl_data, split,
                           TARGET_LEN, N_PV, scenario_label: int):
    """
    Generate trajectories for all windows in split, forcing scenario_label.
    Returns (N, TARGET_LEN, N_PV) generated PV trajectories.
    """
    N    = len(X)
    out  = np.zeros((N, TARGET_LEN, N_PV), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl        = slice(i, i + BATCH)
            x_cv_b    = torch.tensor(X[sl]).float().to(DEVICE)
            xct_b     = torch.tensor(X_cv_tgt[sl]).float().to(DEVICE).clone()
            pv_init_b = torch.tensor(pv_init[sl]).float().to(DEVICE)
            # Force the scenario label for ALL windows
            sc_b      = torch.full((x_cv_b.size(0),), scenario_label,
                                   dtype=torch.long, device=DEVICE)
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_col_idx: continue
                Xc      = torch.tensor(ctrl_data[ln][f'X_{split}'][sl]).float().to(DEVICE)
                cv_pred = ctrl_models[ln].predict(Xc, target_len=TARGET_LEN)
                ci      = ctrl_cv_col_idx[ln]
                xct_b[:, :, ci:ci+1] = cv_pred
            pv_seq, _ = plant_model.predict(x_cv_b, xct_b, pv_init_b, sc_b)
            out[i:i + x_cv_b.size(0)] = pv_seq.cpu().numpy()
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main(ckpt_path: Path, n_samples: int):
    print("=" * 60)
    print("HAI Digital Twin — Synthetic Data Generator")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    data        = load_and_prepare_data()
    plant_data  = data['plant']
    ctrl_data   = data['ctrl']
    sensor_cols = data['metadata']['sensor_cols']
    TARGET_LEN  = data['metadata']['target_len']

    # Augment ctrl data
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

    # ── Pick normal seed windows ───────────────────────────────────────────────
    # Use normal training windows as seeds — we generate attack variants from them
    normal_mask = (plant_data['scenario_train'] == 0)
    normal_idx  = np.where(normal_mask)[0]
    np.random.seed(42)
    seed_idx    = np.random.choice(normal_idx, size=min(n_samples, len(normal_idx)),
                                   replace=False)

    X_seed      = plant_data['X_train'][seed_idx]
    X_cv_seed   = plant_data['X_cv_target_train'][seed_idx]
    pv_init_seed = plant_data['pv_init_train'][seed_idx]
    # build a seed ctrl array per loop
    ctrl_data_seed = {}
    for ln in CTRL_LOOPS:
        ctrl_data_seed[ln] = {'X_seed': ctrl_data[ln]['X_train'][seed_idx]}

    print(f"\n[3] Generating {len(seed_idx)} synthetic windows per scenario...")

    generated = {}
    for sc_id, sc_name in SCENARIO_NAMES.items():
        print(f"    Generating {sc_name}...")
        gen = generate_for_scenario(
            plant_model, ctrl_models, ctrl_cv_col_idx,
            X_seed, X_cv_seed, pv_init_seed,
            ctrl_data_seed, 'seed', TARGET_LEN, N_PV, sc_id
        )
        generated[sc_id] = gen

    # ── Plot 1: Side-by-side scenario comparison (single seed window) ──────────
    print("\n[4] Plotting scenario comparison...")
    seed_w = 0  # first seed window
    fig1, axes1 = plt.subplots(N_PV, len(SCENARIO_NAMES),
                               figsize=(4 * len(SCENARIO_NAMES), 3 * N_PV),
                               sharey='row', squeeze=False)
    t_ax = np.arange(TARGET_LEN)
    for col, (sc_id, sc_name) in enumerate(SCENARIO_NAMES.items()):
        for row, pv_name in enumerate(PV_COLS):
            ax = axes1[row, col]
            ax.plot(t_ax, generated[sc_id][seed_w, :, row],
                    color=SCENARIO_COLORS[sc_id], linewidth=1.5)
            if col == 0:
                ax.set_ylabel(pv_name, fontsize=8)
            if row == 0:
                ax.set_title(sc_name, fontsize=9, color=SCENARIO_COLORS[sc_id],
                             fontweight='bold')
            ax.tick_params(labelsize=6)
            ax.set_xlabel("t (s)", fontsize=7)
    fig1.suptitle("Synthetic trajectories — same seed window, all 4 scenarios",
                  fontsize=11)
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "scenario_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"    Saved: {OUT_DIR}/scenario_comparison.png")

    # ── Plot 2: Generated vs Real for each attack scenario ─────────────────────
    print("\n[5] Plotting generated vs real per attack scenario...")
    real_test    = plant_data['pv_target_test']
    scenario_test = plant_data['scenario_test']

    for sc_id in [1, 2, 3]:
        sc_name  = SCENARIO_NAMES[sc_id]
        real_mask = np.where(scenario_test == sc_id)[0]
        if len(real_mask) == 0:
            continue

        # mean trajectory: real vs generated
        real_mean = real_test[real_mask].mean(axis=0)          # (T, N_PV)
        gen_mean  = generated[sc_id].mean(axis=0)              # (T, N_PV)

        fig2, axes2 = plt.subplots(1, N_PV, figsize=(3.5 * N_PV, 3.5), squeeze=False)
        for k, pv_name in enumerate(PV_COLS):
            ax = axes2[0, k]
            ax.plot(t_ax, real_mean[:, k],
                    color=SCENARIO_COLORS[sc_id], linewidth=2, label='Real')
            ax.plot(t_ax, gen_mean[:, k],
                    color='black', linewidth=1.4, linestyle='--', label='Synthetic')
            ax.fill_between(t_ax, real_mean[:, k], gen_mean[:, k],
                            alpha=0.15, color=SCENARIO_COLORS[sc_id])
            ax.set_title(pv_name, fontsize=8)
            ax.set_xlabel("t (s)", fontsize=7)
            ax.tick_params(labelsize=6)
            if k == 0:
                ax.set_ylabel("Scaled value", fontsize=7)
                ax.legend(fontsize=7)
        fig2.suptitle(f"Real vs Synthetic — {sc_name} (mean over all windows)",
                      fontsize=10, color=SCENARIO_COLORS[sc_id])
        fig2.tight_layout()
        fig2.savefig(OUT_DIR / f"real_vs_synthetic_{sc_name}.png",
                     dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"    Saved: {OUT_DIR}/real_vs_synthetic_{sc_name}.png")

    # ── Plot 3: Distribution comparison (mean abs per window) ─────────────────
    print("\n[6] Plotting distribution comparison...")
    fig3, axes3 = plt.subplots(1, N_PV, figsize=(3.2 * N_PV, 3.5), squeeze=False)
    for k, pv_name in enumerate(PV_COLS):
        ax = axes3[0, k]
        for sc_id, sc_name in SCENARIO_NAMES.items():
            vals = generated[sc_id][:, :, k].mean(axis=1)  # mean per window
            ax.hist(vals, bins=30, alpha=0.5, color=SCENARIO_COLORS[sc_id],
                    label=sc_name, density=True)
        ax.set_title(pv_name, fontsize=8)
        ax.set_xlabel("Mean generated value", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("Density", fontsize=7)
            ax.legend(fontsize=6)
    fig3.suptitle("Distribution of generated trajectories per scenario",
                  fontsize=10)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "generation_distributions.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"    Saved: {OUT_DIR}/generation_distributions.png")

    # ── Plot 4: PCA of generated trajectories ─────────────────────────────────
    print("\n[7] PCA of generated trajectories...")
    # flatten each window to a vector: (N, TARGET_LEN * N_PV)
    all_gen   = np.concatenate([generated[sc_id] for sc_id in SCENARIO_NAMES], axis=0)
    all_flat  = all_gen.reshape(len(all_gen), -1)
    all_labels = np.concatenate([
        np.full(len(generated[sc_id]), sc_id) for sc_id in SCENARIO_NAMES
    ])

    scaler_pca = StandardScaler()
    all_flat_s = scaler_pca.fit_transform(all_flat)
    pca        = PCA(n_components=2, random_state=42)
    coords     = pca.fit_transform(all_flat_s)

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    for sc_id, sc_name in SCENARIO_NAMES.items():
        mask = (all_labels == sc_id)
        ax4.scatter(coords[mask, 0], coords[mask, 1],
                    color=SCENARIO_COLORS[sc_id], label=sc_name,
                    alpha=0.4, s=8)
    ax4.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=9)
    ax4.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=9)
    ax4.set_title("PCA of synthetic trajectories — scenario separation", fontsize=10)
    ax4.legend(fontsize=8)
    fig4.tight_layout()
    fig4.savefig(OUT_DIR / "pca_generated.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"    Saved: {OUT_DIR}/pca_generated.png")

    # ── Plot 5: NRMSE between generated and real per scenario ─────────────────
    print("\n[8] Computing generation quality (NRMSE vs real)...")
    nrmse_results = {}
    for sc_id in [1, 2, 3]:
        sc_name   = SCENARIO_NAMES[sc_id]
        real_mask = np.where(scenario_test == sc_id)[0]
        if len(real_mask) == 0:
            continue
        real_sc = real_test[real_mask]
        gen_sc  = generated[sc_id][:len(real_sc)]
        nrmse_per_pv = []
        for k in range(N_PV):
            r   = real_sc[:, :, k]
            g   = gen_sc[:, :, k]
            rng = r.max() - r.min() + 1e-8
            nrmse_per_pv.append(float(np.sqrt(((r - g) ** 2).mean()) / rng))
        nrmse_results[sc_name] = nrmse_per_pv
        print(f"    {sc_name}: NRMSE per PV = "
              + "  ".join(f"{v:.4f}" for v in nrmse_per_pv))

    fig5, ax5 = plt.subplots(figsize=(8, 3.5))
    x = np.arange(N_PV)
    width = 0.25
    for i, (sc_name, nrmse_vals) in enumerate(nrmse_results.items()):
        sc_id = [k for k, v in SCENARIO_NAMES.items() if v == sc_name][0]
        ax5.bar(x + i * width, nrmse_vals, width,
                label=sc_name, color=SCENARIO_COLORS[sc_id], alpha=0.85)
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(PV_COLS, fontsize=8)
    ax5.set_ylabel("NRMSE", fontsize=9)
    ax5.set_title("Generation quality — NRMSE between synthetic and real per scenario",
                  fontsize=10)
    ax5.legend(fontsize=8)
    fig5.tight_layout()
    fig5.savefig(OUT_DIR / "generation_nrmse.png", dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"    Saved: {OUT_DIR}/generation_nrmse.png")

    # ── Save synthetic dataset ─────────────────────────────────────────────────
    print("\n[9] Saving synthetic dataset...")
    for sc_id, sc_name in SCENARIO_NAMES.items():
        np.save(OUT_DIR / f"synthetic_{sc_name}.npy", generated[sc_id])
        print(f"    Saved: {OUT_DIR}/synthetic_{sc_name}.npy  "
              f"shape={generated[sc_id].shape}")

    summary = {
        "checkpoint":   str(ckpt_path),
        "n_seeds":      int(len(seed_idx)),
        "target_len":   int(TARGET_LEN),
        "n_pv":         int(N_PV),
        "pv_cols":      PV_COLS,
        "nrmse_vs_real": nrmse_results,
    }
    with open(OUT_DIR / "generation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: {OUT_DIR}/generation_results.json")

    print("\nDone.")
    print(f"\nOutputs in: {OUT_DIR}/")
    print("  scenario_comparison.png   — same seed, 4 scenarios side by side")
    print("  real_vs_synthetic_*.png   — real vs generated mean trajectories")
    print("  generation_distributions.png — value distributions per scenario")
    print("  pca_generated.png         — PCA showing scenario separation")
    print("  generation_nrmse.png      — generation quality per scenario/PV")
    print("  synthetic_*.npy           — synthetic data arrays (ready to use)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/pipeline/gru_scenario_weighted/gru_plant.pt")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of normal seed windows to generate from")
    args = parser.parse_args()
    main(Path(args.ckpt), args.n_samples)
