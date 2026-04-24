"""
augment_idea1_all_seeds.py — AE augmentation via all-scenario seeds.

Uses ALL available test windows (Normal + AP_no + AP_with) as input seeds
but forces scenario label = 3 (AE_no) during generation. This gives
thousands of diverse input contexts for AE trajectory generation.

Diversity source: 3292 normal + 42 AP_no + 37 AP_with = ~3371 seeds
                  vs only 6 real AE_no seeds in Idea 3.

Usage:
    python 05_detect/augment_idea1_all_seeds.py
    python 05_detect/augment_idea1_all_seeds.py --n_synth 300 --ckpt outputs/pipeline/gru_scenario_haiend/gru_plant.pt
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "02_data_pipeline"))
sys.path.insert(0, str(ROOT / "03_model"))

from pipeline import load_and_prepare_data
from gru import GRUPlant, GRUController, CCSequenceModel
from config import LOOPS, PV_COLS, PROCESSED_DATA_DIR

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 128
CTRL_LOOPS = ['PC', 'LC', 'FC', 'TC', 'CC']
SCENARIO_NAMES  = {0: "Normal", 1: "AP_no", 2: "AP_with", 3: "AE_no"}
SCENARIO_COLORS = {0: "#2196F3", 1: "#FF5722", 2: "#E91E63", 3: "#9C27B0"}
OUT_DIR = ROOT / "outputs" / "augment_idea1_all_seeds"
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
    """Generate PV trajectories. sc_label forces scenario label for ALL windows."""
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


def extract_features(traj):
    feats = []
    for k in range(traj.shape[-1]):
        r = traj[:, :, k]
        feats += [r.mean(1), r.std(1), r.min(1), r.max(1),
                  np.abs(r).mean(1), np.diff(r, axis=1).mean(1)]
    return np.stack(feats, axis=1)


def run_clf(Xtr, ytr, Xte, yte, label):
    sc  = StandardScaler()
    clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                 class_weight='balanced')
    clf.fit(sc.fit_transform(Xtr), ytr)
    yp  = clf.predict(sc.transform(Xte))
    tnames = [SCENARIO_NAMES[i] for i in sorted(set(ytr) | set(yte))]
    print(f"\n  [{label}]")
    print(classification_report(yte, yp, target_names=tnames, zero_division=0))
    return f1_score(yte, yp, average='macro', zero_division=0)


def main(ckpt_path, n_synth):
    print("=" * 60)
    print("Idea 1 — AE Augmentation via All-Scenario Seeds")
    print(f"  n_synth={n_synth}  (AE label forced on all seeds)")
    print("=" * 60)

    data       = load_and_prepare_data()
    plant_data = data['plant']
    ctrl_data  = data['ctrl']
    sensor_cols = data['metadata']['sensor_cols']
    TARGET_LEN  = data['metadata']['target_len']

    # augment ctrl data with extra channels
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

    # ── Pool: ALL non-AE test windows as seeds ────────────────────────────────
    non_ae_mask = np.where(plant_data['scenario_test'] != 3)[0]
    print(f"\n  Non-AE seed pool: {len(non_ae_mask)} windows")
    for sc_id in [0, 1, 2]:
        n = (plant_data['scenario_test'][non_ae_mask] == sc_id).sum()
        print(f"    {SCENARIO_NAMES[sc_id]}: {n}")

    # Sample n_synth seeds from pool
    np.random.seed(42)
    chosen = np.random.choice(non_ae_mask, size=n_synth, replace=(len(non_ae_mask) < n_synth))

    X_seeds   = plant_data['X_test'][chosen]
    Xcv_seeds = plant_data['X_cv_target_test'][chosen]
    pvi_seeds = plant_data['pv_init_test'][chosen]
    ctrl_seeds = {ln: ctrl_data[ln]['X_test'][chosen] for ln in CTRL_LOOPS}

    print(f"\n  Generating {n_synth} AE_no trajectories from non-AE seeds...")
    ae_synth = generate(plant, ctrls, ctrl_cv_idx,
                        X_seeds, Xcv_seeds, pvi_seeds, ctrl_seeds,
                        TARGET_LEN, N_PV, sc_label=3)
    print(f"  Generated: {ae_synth.shape}")

    # ── Also generate other scenarios balanced to same n_synth ───────────────
    all_synth = {3: ae_synth}
    for sc_id in [0, 1, 2]:
        mask = np.where(plant_data['scenario_test'] == sc_id)[0]
        if len(mask) == 0: continue
        rep  = np.random.choice(mask, size=n_synth, replace=True)
        all_synth[sc_id] = generate(
            plant, ctrls, ctrl_cv_idx,
            plant_data['X_test'][rep],
            plant_data['X_cv_target_test'][rep],
            plant_data['pv_init_test'][rep],
            {ln: ctrl_data[ln]['X_test'][rep] for ln in CTRL_LOOPS},
            TARGET_LEN, N_PV, sc_id
        )

    X_synth = np.concatenate([all_synth[i] for i in sorted(all_synth)], axis=0)
    y_synth = np.concatenate([np.full(n_synth, i) for i in sorted(all_synth)])

    pv_real  = plant_data['pv_target_test']
    y_real   = plant_data['scenario_test']
    pv_train = plant_data['pv_target_train']
    y_train  = plant_data['scenario_train']

    Xrf = extract_features(pv_real)
    Xsf = extract_features(X_synth)
    Xtf = extract_features(pv_train)

    print("\n" + "=" * 50)
    print("Exp A: Train REAL → Test SYNTHETIC")
    f1_A = run_clf(Xrf, y_real, Xsf, y_synth, "Real→Synth")

    print("\n" + "=" * 50)
    print("Exp B: Train SYNTHETIC → Test REAL")
    f1_B = run_clf(Xsf, y_synth, Xrf, y_real, "Synth→Real")

    print("\n" + "=" * 50)
    print("Exp C: Train REAL(train)+SYNTHETIC → Test REAL(test)")
    Xm = np.concatenate([Xtf, Xsf], axis=0)
    ym = np.concatenate([y_train, y_synth])
    f1_C = run_clf(Xm, ym, Xrf, y_real, "Mixed→Real")

    # ── Compare AE real seeds vs idea1 synthetic ──────────────────────────────
    ae_real_mean   = pv_real[y_real == 3].mean(axis=0)
    ae_synth_mean  = ae_synth.mean(axis=0)
    t = np.arange(TARGET_LEN)

    fig, axes = plt.subplots(1, N_PV, figsize=(3.5*N_PV, 3.5), squeeze=False)
    for k, pv in enumerate(PV_COLS):
        ax = axes[0, k]
        ax.plot(t, ae_real_mean[:, k],  color='#9C27B0', lw=2, label='Real AE_no')
        ax.plot(t, ae_synth_mean[:, k], color='black',   lw=1.4,
                linestyle='--', label='Synthetic (idea1)')
        ax.fill_between(t, ae_real_mean[:, k], ae_synth_mean[:, k],
                        alpha=0.15, color='#9C27B0')
        ax.set_title(pv, fontsize=8)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("Scaled value", fontsize=7)
            ax.legend(fontsize=6)
    fig.suptitle(f"Idea 1 — AE_no: Real vs All-Seed Synthetic\n"
                 f"({len(non_ae_mask)} non-AE seeds → {n_synth} AE trajectories)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ae_real_vs_synthetic.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Seed breakdown plot ───────────────────────────────────────────────────
    seed_sc = plant_data['scenario_test'][chosen]
    sc_counts = {sc_id: (seed_sc == sc_id).sum() for sc_id in [0, 1, 2]}
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    bars = ax2.bar([SCENARIO_NAMES[i] for i in [0, 1, 2]],
                   [sc_counts[i] for i in [0, 1, 2]],
                   color=[SCENARIO_COLORS[i] for i in [0, 1, 2]])
    ax2.set_title(f"Seed scenario breakdown (n_synth={n_synth})", fontsize=10)
    ax2.set_ylabel("Count")
    for bar, sc_id in zip(bars, [0, 1, 2]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(sc_counts[sc_id]), ha='center', fontsize=9)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "seed_breakdown.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"\n{'='*50}")
    print(f"SUMMARY — Idea 1 (All-Scenario Seeds, n_synth={n_synth})")
    print(f"  Exp A (Real→Synth):  {f1_A:.4f}")
    print(f"  Exp B (Synth→Real):  {f1_B:.4f}")
    print(f"  Exp C (Mixed→Real):  {f1_C:.4f}")
    print(f"  Saved: {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/pipeline/gru_scenario_haiend/gru_plant.pt")
    parser.add_argument("--n_synth", type=int, default=300)
    args = parser.parse_args()
    main(ROOT / args.ckpt, args.n_synth)
