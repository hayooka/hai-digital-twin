"""
augment_idea2_latent.py — AE augmentation via latent space interpolation.

Encodes all 6 real AE_no windows through the GRUPlant encoder to get
their hidden states. Linearly interpolates between every pair of hidden
states at multiple alpha values, then decodes each interpolated state
into a new AE trajectory.

Diversity source: C(6,2) × n_alphas = 15 × 20 = 300 unique interpolants
                  that span the AE-conditioned latent subspace.

Key: the decoder input (x_cv_target, pv_init) is averaged from the two
     endpoint windows, giving a natural conditioning for the interpolated
     trajectory.

Usage:
    python 05_detect/augment_idea2_latent.py
    python 05_detect/augment_idea2_latent.py --n_alphas 20
"""

import sys
import argparse
import numpy as np
import torch
import joblib
from pathlib import Path
from itertools import combinations
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
OUT_DIR = ROOT / "outputs" / "augment_idea2_latent"
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


def encode_ae_seeds(plant, X_ae, Xcv_ae):
    """
    Run encoder on AE seeds and return hidden states.
    Returns h: (N_seeds, layers, hidden)
    """
    hidden_states = []
    with torch.no_grad():
        for i in range(len(X_ae)):
            xb = torch.tensor(X_ae[i:i+1]).float().to(DEVICE)
            sc = torch.tensor([3], dtype=torch.long, device=DEVICE)
            emb = plant.scenario_emb(sc).unsqueeze(1).expand(-1, xb.size(1), -1)
            _, h = plant.encoder(torch.cat([xb, emb], dim=-1))  # (layers, 1, hidden)
            hidden_states.append(h.squeeze(1).cpu().numpy())    # (layers, hidden)
    return np.stack(hidden_states, axis=0)                       # (N, layers, hidden)


def decode_from_hidden(plant, ctrls, ctrl_cv_idx, h_interp,
                       Xcv_tgt, pvi, ctrl_seed, TL, NP):
    """
    Decode PV trajectories from pre-computed hidden states h_interp.
    h_interp: (N, layers, hidden) numpy array
    """
    N   = len(h_interp)
    out = np.zeros((N, TL, NP), dtype=np.float32)
    sc_label = 3  # always AE

    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl  = slice(i, i + BATCH)
            bs  = min(BATCH, N - i)

            # Build CV target with controller predictions
            xcb = torch.tensor(Xcv_tgt[sl]).float().to(DEVICE).clone()
            for ln in CTRL_LOOPS:
                if ln not in ctrl_cv_idx: continue
                Xc = torch.tensor(ctrl_seed[ln][sl]).float().to(DEVICE)
                cp = ctrls[ln].predict(Xc, target_len=TL)
                xcb[:, :, ctrl_cv_idx[ln]:ctrl_cv_idx[ln]+1] = cp

            pb = torch.tensor(pvi[sl]).float().to(DEVICE)
            h  = torch.tensor(h_interp[sl]).float().to(DEVICE)
            # h needs shape (layers, B, hidden)
            h  = h.permute(1, 0, 2).contiguous()

            sc = torch.full((bs,), sc_label, dtype=torch.long, device=DEVICE)

            # Decode autoregressively from interpolated h
            pv = pb
            pv_outputs = []
            for t in range(TL):
                dec_in = torch.cat([xcb[:, t, :], pv], dim=-1).unsqueeze(1)
                dec_out, h = plant.decoder(dec_in, h)
                h_out = dec_out.squeeze(1)
                if plant.scenario_heads:
                    pv_pred = torch.zeros(bs, plant.n_pv, device=DEVICE)
                    for sc_id in range(plant.n_scenarios):
                        mask = (sc == sc_id)
                        if mask.any():
                            pv_pred[mask] = plant.fc_heads[sc_id](h_out[mask])
                else:
                    pv_pred = plant.fc(h_out)
                pv_outputs.append(pv_pred)
                pv = pv_pred
            out[i:i+bs] = torch.stack(pv_outputs, dim=1).cpu().numpy()
    return out


def generate_other_scenarios(plant, ctrls, ctrl_cv_idx, plant_data, ctrl_data,
                              TARGET_LEN, N_PV, n_synth):
    """Generate n_synth trajectories for each non-AE scenario."""
    all_synth = {}
    np.random.seed(42)
    for sc_id in [0, 1, 2]:
        mask = np.where(plant_data['scenario_test'] == sc_id)[0]
        if len(mask) == 0: continue
        rep = np.random.choice(mask, size=n_synth, replace=True)
        out = np.zeros((n_synth, TARGET_LEN, N_PV), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n_synth, BATCH):
                sl  = slice(i, i + BATCH)
                xb  = torch.tensor(plant_data['X_test'][rep[sl]]).float().to(DEVICE)
                xcb = torch.tensor(plant_data['X_cv_target_test'][rep[sl]]).float().to(DEVICE).clone()
                pb  = torch.tensor(plant_data['pv_init_test'][rep[sl]]).float().to(DEVICE)
                sb  = torch.full((xb.size(0),), sc_id, dtype=torch.long, device=DEVICE)
                for ln in CTRL_LOOPS:
                    if ln not in ctrl_cv_idx: continue
                    Xc = torch.tensor(ctrl_data[ln]['X_test'][rep[sl]]).float().to(DEVICE)
                    cp = ctrls[ln].predict(Xc, target_len=TARGET_LEN)
                    xcb[:, :, ctrl_cv_idx[ln]:ctrl_cv_idx[ln]+1] = cp
                pv, _ = plant.predict(xb, xcb, pb, sb)
                out[i:i+xb.size(0)] = pv.cpu().numpy()
        all_synth[sc_id] = out
    return all_synth


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


def main(ckpt_path, n_alphas):
    print("=" * 60)
    print("Idea 2 — AE Augmentation via Latent Space Interpolation")
    print(f"  n_alphas={n_alphas}  (interpolation steps between each pair)")
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

    # ── Encode 6 real AE_no seeds ─────────────────────────────────────────────
    ae_mask = np.where(plant_data['scenario_test'] == 3)[0]
    print(f"\n  Real AE_no seeds: {len(ae_mask)} windows")

    X_ae    = plant_data['X_test'][ae_mask]
    Xcv_ae  = plant_data['X_cv_target_test'][ae_mask]
    pvi_ae  = plant_data['pv_init_test'][ae_mask]
    ctrl_ae = {ln: ctrl_data[ln]['X_test'][ae_mask] for ln in CTRL_LOOPS}

    print("  Encoding AE seeds to hidden states...")
    h_seeds = encode_ae_seeds(plant, X_ae, Xcv_ae)   # (6, layers, hidden)
    print(f"  Hidden state shape per seed: {h_seeds.shape[1:]}")

    # ── Interpolate between all pairs ─────────────────────────────────────────
    pairs = list(combinations(range(len(ae_mask)), 2))
    alphas = np.linspace(0.0, 1.0, n_alphas + 2)[1:-1]  # exclude endpoints (real seeds)
    print(f"\n  Pairs: {len(pairs)}  alphas: {len(alphas)}  "
          f"→ {len(pairs)*len(alphas)} interpolants")

    h_interp_list   = []
    Xcv_interp_list = []
    pvi_interp_list = []
    ctrl_interp     = {ln: [] for ln in CTRL_LOOPS}

    for i, j in pairs:
        for alpha in alphas:
            h_ij = (1 - alpha) * h_seeds[i] + alpha * h_seeds[j]
            h_interp_list.append(h_ij)
            # Decoder conditioning: average of the two endpoint windows
            Xcv_interp_list.append(
                (1 - alpha) * Xcv_ae[i] + alpha * Xcv_ae[j])
            pvi_interp_list.append(
                (1 - alpha) * pvi_ae[i] + alpha * pvi_ae[j])
            for ln in CTRL_LOOPS:
                ctrl_interp[ln].append(
                    (1 - alpha) * ctrl_ae[ln][i] + alpha * ctrl_ae[ln][j])

    h_interp   = np.stack(h_interp_list,   axis=0)  # (N, layers, hidden)
    Xcv_interp = np.stack(Xcv_interp_list, axis=0)
    pvi_interp = np.stack(pvi_interp_list, axis=0)
    ctrl_interp_np = {ln: np.stack(ctrl_interp[ln], axis=0) for ln in CTRL_LOOPS}

    n_interp = len(h_interp)
    print(f"  Decoding {n_interp} interpolated hidden states...")
    ae_synth = decode_from_hidden(
        plant, ctrls, ctrl_cv_idx,
        h_interp, Xcv_interp, pvi_interp, ctrl_interp_np,
        TARGET_LEN, N_PV
    )
    print(f"  Generated: {ae_synth.shape}")

    # ── Generate other scenarios (balanced) ───────────────────────────────────
    all_synth = generate_other_scenarios(
        plant, ctrls, ctrl_cv_idx, plant_data, ctrl_data,
        TARGET_LEN, N_PV, n_synth=n_interp
    )
    all_synth[3] = ae_synth

    X_synth = np.concatenate([all_synth[i] for i in sorted(all_synth)], axis=0)
    y_synth = np.concatenate([np.full(len(all_synth[i]), i)
                               for i in sorted(all_synth)])

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

    # ── Plot: interpolation trajectory ────────────────────────────────────────
    # Show endpoint A, endpoint B, and several interpolants for PV[0]
    fig, axes = plt.subplots(1, N_PV, figsize=(3.5*N_PV, 3.5), squeeze=False)
    ae_real_mean   = pv_real[y_real == 3].mean(axis=0)
    ae_synth_mean  = ae_synth.mean(axis=0)
    t = np.arange(TARGET_LEN)

    # Show a few individual interpolants for first pair
    n_show = min(5, len(alphas))
    show_idx = [i for i, a in enumerate(alphas) if i < n_show]

    for k, pv in enumerate(PV_COLS):
        ax = axes[0, k]
        ax.plot(t, ae_real_mean[:, k],  color='#9C27B0', lw=2,
                label='Real AE_no mean', zorder=3)
        ax.plot(t, ae_synth_mean[:, k], color='black',   lw=1.4,
                linestyle='--', label='Synth mean', zorder=3)
        # Show a sample interpolant
        for idx in show_idx:
            ax.plot(t, ae_synth[idx, :, k], color='#9C27B0', lw=0.5,
                    alpha=0.3, zorder=1)
        ax.set_title(pv, fontsize=8)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("Scaled value", fontsize=7)
            ax.legend(fontsize=6)
    fig.suptitle(f"Idea 2 — AE_no: Real vs Latent Interpolation Synthetic\n"
                 f"({len(pairs)} pairs × {len(alphas)} alphas = {n_interp} trajectories)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ae_real_vs_synthetic.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot: hidden state PCA ────────────────────────────────────────────────
    from sklearn.decomposition import PCA
    # Flatten hidden states: (N, layers*hidden)
    h_real_flat  = h_seeds.reshape(len(ae_mask), -1)
    h_interp_flat = h_interp.reshape(n_interp, -1)
    all_h = np.concatenate([h_real_flat, h_interp_flat], axis=0)
    pca2  = PCA(n_components=2).fit(all_h)
    hr_2d = pca2.transform(h_real_flat)
    hi_2d = pca2.transform(h_interp_flat)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.scatter(hi_2d[:, 0], hi_2d[:, 1], s=4, alpha=0.3,
                color='#9C27B0', label='Interpolated')
    ax2.scatter(hr_2d[:, 0], hr_2d[:, 1], s=60, color='black',
                marker='*', zorder=5, label='Real AE seeds')
    ax2.set_title("Latent space (PCA 2D) — Real AE seeds + interpolants")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "latent_pca.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"\n{'='*50}")
    print(f"SUMMARY — Idea 2 (Latent Interpolation, n_alphas={n_alphas})")
    print(f"  Exp A (Real→Synth):  {f1_A:.4f}")
    print(f"  Exp B (Synth→Real):  {f1_B:.4f}")
    print(f"  Exp C (Mixed→Real):  {f1_C:.4f}")
    print(f"  Saved: {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/pipeline/gru_scenario_haiend/gru_plant.pt")
    parser.add_argument("--n_alphas", type=int, default=20)
    args = parser.parse_args()
    main(ROOT / args.ckpt, args.n_alphas)
