"""
plot_individual_trajectories.py — Individual trajectory plots for simulation and generation.

Two sets of plots:
  1. SIMULATION  — model predicts what the plant does (predicted vs actual, per scenario)
  2. GENERATION  — model generates diverse synthetic trajectories (multiple samples overlay)

Usage:
    python 05_detect/plot_individual_trajectories.py
"""

import sys
import argparse
import numpy as np
import torch
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 128
CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC', 'CC']

SCENARIO_NAMES  = {0: "Normal", 1: "AP\_no", 2: "AP\_with", 3: "AE\_no"}
SCENARIO_COLORS = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}

OUT_DIR = ROOT / "outputs" / "individual_trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_CHANNELS = {
    'PC': ['P1_PCV02D', 'P1_FT01',   'P1_TIT01'],
    'LC': ['P1_FT03',   'P1_FCV03D', 'P1_PCV01D'],
    'FC': ['P1_PIT01',  'P1_LIT01',  'P1_TIT03'],
    'TC': ['P1_FT02',   'P1_PIT02',  'P1_TIT02'],
    'CC': ['P1_PP04D',  'P1_FCV03D', 'P1_PCV02D'],
}


def load_models(ckpt_path, data):
    pd   = data['plant']
    cd   = data['ctrl']
    TL   = data['metadata']['target_len']
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    hidden   = ckpt.get('hidden', 512)
    layers   = ckpt.get('layers', 2)
    n_haiend = ckpt.get('n_haiend', pd['n_haiend'])

    plant = GRUPlant(n_plant_in=pd['n_plant_in'], n_pv=pd['n_pv'],
                     hidden=hidden, layers=layers,
                     n_scenarios=data['metadata']['n_scenarios'],
                     dropout=0.0, n_haiend=n_haiend).to(DEVICE)
    plant.load_state_dict(ckpt['model_state'], strict=False)
    plant.eval()

    CTRL_H = {'PC': 64, 'LC': 64, 'FC': 128, 'TC': 64, 'CC': 64}
    ctrls  = {}
    for ln in CTRL_LOOPS:
        n_in = cd[ln]['X_train'].shape[-1]
        m = CCSequenceModel(n_inputs=n_in, hidden=CTRL_H[ln], layers=2,
                            dropout=0.0, output_len=TL).to(DEVICE) \
            if ln == 'CC' else \
            GRUController(n_inputs=n_in, hidden=CTRL_H[ln], layers=2,
                          dropout=0.0, output_len=TL).to(DEVICE)
        p = ckpt_path.parent / f"gru_ctrl_{ln.lower()}.pt"
        if p.exists():
            c = torch.load(p, map_location=DEVICE)
            m.load_state_dict(c['model_state'], strict=False)
        m.eval()
        ctrls[ln] = m
    return plant, ctrls, TL, pd['n_pv']


def generate(plant, ctrls, ctrl_cv_idx, X, Xcv, pvi, ctrl_seed, TL, NP, sc_label):
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
                Xc = torch.tensor(ctrl_seed[ln][sl]).float().to(DEVICE)
                cp = ctrls[ln].predict(Xc, target_len=TL)
                xcb[:, :, ctrl_cv_idx[ln]:ctrl_cv_idx[ln]+1] = cp
            pv, _ = plant.predict(xb, xcb, pb, sb)
            out[i:i+xb.size(0)] = pv.cpu().numpy()
    return out


def plot_simulation(sc_id, real_traj, pred_traj, TARGET_LEN, color, label):
    """
    Simulation plot: predicted vs actual for one scenario.
    Picks the window with highest variance (most interesting window).
    Shows all 5 PVs.
    """
    # pick most interesting window (highest variance across PVs)
    var = real_traj.var(axis=(1, 2))
    idx = np.argsort(var)[::-1][:5]  # top 5 windows

    t = np.arange(TARGET_LEN)
    fig, axes = plt.subplots(len(idx), len(PV_COLS),
                              figsize=(3.2 * len(PV_COLS), 2.5 * len(idx)),
                              squeeze=False)

    for row, wi in enumerate(idx):
        for col, pv in enumerate(PV_COLS):
            ax = axes[row, col]
            ax.plot(t, real_traj[wi, :, col], color='black', lw=1.5,
                    label='Actual' if row == 0 else '')
            ax.plot(t, pred_traj[wi, :, col], color=color, lw=1.5,
                    linestyle='--', label='Predicted' if row == 0 else '')
            ax.fill_between(t,
                            real_traj[wi, :, col],
                            pred_traj[wi, :, col],
                            alpha=0.15, color=color)
            if row == 0:
                ax.set_title(pv, fontsize=8, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"Window {wi}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("t (s)", fontsize=6)

    handles = [
        plt.Line2D([0], [0], color='black', lw=1.5, label='Actual'),
        plt.Line2D([0], [0], color=color, lw=1.5, linestyle='--', label='Predicted'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
    fig.suptitle(f"Simulation — {label}: Predicted vs Actual (top 5 windows by variance)",
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout()
    fname = OUT_DIR / f"simulation_{label.replace('_', '')}.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_generation(sc_id, synth_traj, real_traj, TARGET_LEN, color, label, n_samples=8):
    """
    Generation plot: overlay of multiple synthetic trajectories per scenario.
    Shows diversity of generation + mean vs real mean.
    """
    t = np.arange(TARGET_LEN)
    np.random.seed(42)
    sample_idx = np.random.choice(len(synth_traj), size=min(n_samples, len(synth_traj)),
                                   replace=False)

    real_mean  = real_traj.mean(axis=0)
    synth_mean = synth_traj.mean(axis=0)

    fig, axes = plt.subplots(1, len(PV_COLS),
                              figsize=(3.2 * len(PV_COLS), 3.5),
                              squeeze=False)

    for col, pv in enumerate(PV_COLS):
        ax = axes[0, col]

        # individual synthetic samples (light)
        for si, wi in enumerate(sample_idx):
            ax.plot(t, synth_traj[wi, :, col], color=color, lw=0.6,
                    alpha=0.25, label='Synthetic samples' if si == 0 else '')

        # synthetic mean
        ax.plot(t, synth_mean[:, col], color=color, lw=2.0,
                label='Synthetic mean')

        # real mean
        ax.plot(t, real_mean[:, col], color='black', lw=1.8,
                linestyle='--', label='Real mean')

        ax.set_title(pv, fontsize=8, fontweight='bold')
        ax.set_xlabel("t (s)", fontsize=6)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel("Scaled value", fontsize=7)
            ax.legend(fontsize=6)

    fig.suptitle(f"Generation — {label}: {len(synth_traj)} synthetic trajectories"
                 f" (showing {len(sample_idx)} samples + mean vs real mean)",
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fname = OUT_DIR / f"generation_{label.replace('_', '')}.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def main(ckpt_path, n_synth):
    print("=" * 60)
    print("Individual Trajectory Plots — Simulation & Generation")
    print("=" * 60)

    data        = load_and_prepare_data()
    plant_data  = data['plant']
    ctrl_data   = data['ctrl']
    sensor_cols = data['metadata']['sensor_cols']
    TARGET_LEN  = data['metadata']['target_len']

    ps  = joblib.load(f"{PROCESSED_DATA_DIR}/scaler.pkl")
    npz = {s: np.load(f"{PROCESSED_DATA_DIR}/{s}_data.npz")
           for s in ("train", "val", "test")}
    ci  = {c: i for i, c in enumerate(sensor_cols)}
    for ln, ec_list in EXTRA_CHANNELS.items():
        for ec in ec_list:
            if ec not in ci: continue
            ei = ci[ec]
            me, se = ps.mean_[ei], ps.scale_[ei]
            for sp, arr in npz.items():
                raw = arr['X'][:, :, [ei]].astype(np.float32)
                ctrl_data[ln][f'X_{sp}'] = np.concatenate(
                    [ctrl_data[ln][f'X_{sp}'], (raw - me) / se], axis=-1)

    pv_set = set(PV_COLS)
    npc    = [c for c in sensor_cols if c not in pv_set]
    c2i    = {c: i for i, c in enumerate(npc)}
    ctrl_cv_idx = {ln: c2i[LOOPS[ln].cv]
                   for ln in CTRL_LOOPS if LOOPS[ln].cv in c2i}

    plant, ctrls, TARGET_LEN, N_PV = load_models(ckpt_path, data)

    np.random.seed(42)

    for sc_id in [0, 1, 2, 3]:
        sc_name = SCENARIO_NAMES[sc_id]
        color   = SCENARIO_COLORS[sc_id]
        mask    = np.where(plant_data['scenario_test'] == sc_id)[0]

        if len(mask) == 0:
            print(f"\n  {sc_name}: no test windows — skipping")
            continue

        print(f"\n{'='*50}")
        print(f"  Scenario: {sc_name}  ({len(mask)} real windows)")

        real_traj = plant_data['pv_target_test'][mask]

        # ── Simulation: predict from real inputs, compare to real targets ──
        print(f"  Running simulation (predicted vs actual)...")
        pred_traj = generate(
            plant, ctrls, ctrl_cv_idx,
            plant_data['X_test'][mask],
            plant_data['X_cv_target_test'][mask],
            plant_data['pv_init_test'][mask],
            {ln: ctrl_data[ln]['X_test'][mask] for ln in CTRL_LOOPS},
            TARGET_LEN, N_PV, sc_id  # use true label for simulation
        )
        plot_simulation(sc_id, real_traj, pred_traj, TARGET_LEN, color, sc_name)

        # ── Generation: generate diverse synthetic trajectories ──────────
        n = min(n_synth, len(mask) * 20)
        rep = np.random.choice(mask, size=n, replace=True)
        print(f"  Generating {n} synthetic trajectories...")
        synth_traj = generate(
            plant, ctrls, ctrl_cv_idx,
            plant_data['X_test'][rep],
            plant_data['X_cv_target_test'][rep],
            plant_data['pv_init_test'][rep],
            {ln: ctrl_data[ln]['X_test'][rep] for ln in CTRL_LOOPS},
            TARGET_LEN, N_PV, sc_id
        )
        plot_generation(sc_id, synth_traj, real_traj, TARGET_LEN, color, sc_name)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/pipeline/gru_scenario_haiend/gru_plant.pt")
    parser.add_argument("--n_synth", type=int, default=200)
    args = parser.parse_args()
    main(ROOT / args.ckpt, args.n_synth)
