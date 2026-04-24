"""
augment_idea3_noise.py — AE augmentation via noise injection on real seeds.

Takes the 6 real AE_no windows, adds small Gaussian noise to create
diverse variants, then generates synthetic AE trajectories from each.

Result: 6 seeds × N_variants = diverse AE synthetic windows
        without any retraining.

Usage:
    python 05_detect/augment_idea3_noise.py
    python 05_detect/augment_idea3_noise.py --n_variants 50 --noise_std 0.01
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
OUT_DIR = ROOT / "outputs" / "augment_idea3_noise"
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


def generate(plant, ctrls, ctrl_cv_idx, X, Xcv, pvi, ctrl_seed, TL, NP, sc):
    N   = len(X)
    out = np.zeros((N, TL, NP), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl  = slice(i, i + BATCH)
            xb  = torch.tensor(X[sl]).float().to(DEVICE)
            xcb = torch.tensor(Xcv[sl]).float().to(DEVICE).clone()
            pb  = torch.tensor(pvi[sl]).float().to(DEVICE)
            sb  = torch.full((xb.size(0),), sc, dtype=torch.long, device=DEVICE)
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


def main(ckpt_path, n_variants, noise_std):
    print("=" * 60)
    print("Idea 3 — AE Augmentation via Noise Injection")
    print(f"  n_variants={n_variants}  noise_std={noise_std}")
    print("=" * 60)

    data       = load_and_prepare_data()
    plant_data = data['plant']
    ctrl_data  = data['ctrl']
    sensor_cols = data['metadata']['sensor_cols']
    TARGET_LEN  = data['metadata']['target_len']

    # augment ctrl
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

    # ── Real AE seeds (6 windows) ──────────────────────────────────────────────
    ae_mask = np.where(plant_data['scenario_test'] == 3)[0]
    print(f"\n  Real AE_no seeds: {len(ae_mask)} windows")

    X_ae    = plant_data['X_test'][ae_mask]
    Xcv_ae  = plant_data['X_cv_target_test'][ae_mask]
    pvi_ae  = plant_data['pv_init_test'][ae_mask]
    ctrl_ae = {ln: ctrl_data[ln]['X_test'][ae_mask] for ln in CTRL_LOOPS}

    # ── Noise injection → n_variants copies per seed ──────────────────────────
    print(f"  Creating {len(ae_mask)} × {n_variants} = "
          f"{len(ae_mask)*n_variants} noisy variants...")
    np.random.seed(42)

    X_aug   = np.concatenate([X_ae + np.random.normal(0, noise_std, X_ae.shape)
                               for _ in range(n_variants)], axis=0)
    Xcv_aug = np.concatenate([Xcv_ae + np.random.normal(0, noise_std, Xcv_ae.shape)
                               for _ in range(n_variants)], axis=0)
    pvi_aug = np.concatenate([pvi_ae] * n_variants, axis=0)
    ctrl_aug = {ln: np.concatenate([ctrl_ae[ln]] * n_variants, axis=0)
                for ln in CTRL_LOOPS}

    print(f"  Generating AE_no synthetic trajectories...")
    ae_synth = generate(plant, ctrls, ctrl_cv_idx,
                        X_aug, Xcv_aug, pvi_aug, ctrl_aug,
                        TARGET_LEN, N_PV, 3)
    print(f"  Generated: {ae_synth.shape}")

    # ── Also generate other scenarios from their real seeds ───────────────────
    all_synth = {}
    all_synth[3] = ae_synth
    for sc_id in [0, 1, 2]:
        mask = np.where(plant_data['scenario_test'] == sc_id)[0]
        if len(mask) == 0: continue
        rep  = np.random.choice(mask, size=len(ae_synth), replace=True)
        all_synth[sc_id] = generate(
            plant, ctrls, ctrl_cv_idx,
            plant_data['X_test'][rep],
            plant_data['X_cv_target_test'][rep],
            plant_data['pv_init_test'][rep],
            {ln: ctrl_data[ln]['X_test'][rep] for ln in CTRL_LOOPS},
            TARGET_LEN, N_PV, sc_id
        )

    X_synth = np.concatenate([all_synth[i] for i in sorted(all_synth)], axis=0)
    y_synth = np.concatenate([np.full(len(all_synth[i]), i)
                               for i in sorted(all_synth)])

    pv_real = plant_data['pv_target_test']
    y_real  = plant_data['scenario_test']
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

    # ── Plot: AE real vs synthetic trajectories ───────────────────────────────
    fig, axes = plt.subplots(1, N_PV, figsize=(3.5*N_PV, 3.5), squeeze=False)
    real_ae_mean = pv_real[y_real == 3].mean(axis=0)
    synth_ae_mean = ae_synth.mean(axis=0)
    t = np.arange(TARGET_LEN)
    for k, pv in enumerate(PV_COLS):
        ax = axes[0, k]
        ax.plot(t, real_ae_mean[:, k],  color='#9C27B0', lw=2, label='Real AE_no')
        ax.plot(t, synth_ae_mean[:, k], color='black',   lw=1.4,
                linestyle='--', label='Synthetic (noise)')
        ax.fill_between(t, real_ae_mean[:, k], synth_ae_mean[:, k],
                        alpha=0.15, color='#9C27B0')
        ax.set_title(pv, fontsize=8)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("Scaled value", fontsize=7)
            ax.legend(fontsize=6)
    fig.suptitle(f"Idea 3 — AE_no: Real vs Noise-Augmented Synthetic\n"
                 f"({len(ae_mask)} seeds × {n_variants} variants, σ={noise_std})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ae_real_vs_synthetic.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n{'='*50}")
    print(f"SUMMARY — Idea 3 (Noise Injection, σ={noise_std})")
    print(f"  Exp A (Real→Synth):  {f1_A:.4f}")
    print(f"  Exp B (Synth→Real):  {f1_B:.4f}")
    print(f"  Exp C (Mixed→Real):  {f1_C:.4f}")
    print(f"  Saved: {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/pipeline/gru_scenario_haiend/gru_plant.pt")
    parser.add_argument("--n_variants", type=int, default=50)
    parser.add_argument("--noise_std",  type=float, default=0.01)
    args = parser.parse_args()
    main(ROOT / args.ckpt, args.n_variants, args.noise_std)
