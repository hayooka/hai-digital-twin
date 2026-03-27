"""
LSTM Autoencoder — Anomaly Detection
=====================================
Author  : Bedour Mahdi
Project : Generative Digital Twin for ICS Security
Repo    : hai-digital-twin/models/lstm_autoencoder.py

HOW TO RUN
----------
From the root of the cloned repo:

    python models/lstm_autoencoder.py

Outputs go to:  outputs/bedour/
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed — saves files directly
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0F1117",
    "axes.facecolor":    "#0F1117",
    "savefig.facecolor": "#0F1117",
    "text.color":        "#E8EAF0",
    "axes.labelcolor":   "#E8EAF0",
    "xtick.color":       "#9DA3B0",
    "ytick.color":       "#9DA3B0",
    "axes.edgecolor":    "#2A2D3A",
    "grid.color":        "#2A2D3A",
    "axes.grid":         True,
    "grid.linewidth":    0.5,
    "font.size":         11,
})
ACCENT  = "#4F8EF7"
SUCCESS = "#2ECC71"
DANGER  = "#E74C3C"
WARNING = "#F39C12"
MUTED   = "#9DA3B0"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← all paths and hyper-params in one place
# ─────────────────────────────────────────────────────────────────────────────

# Detect repo root (works whether you run from repo root or models/)
_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(_HERE) if os.path.basename(_HERE) == "models" else _HERE
_DATA     = os.path.join(_REPO, "data")

CFG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "hai_train_glob":          os.path.join(_DATA, "hai-23.05",   "hai-train*.csv"),
    "hai_test_glob":           os.path.join(_DATA, "hai-23.05",   "hai-test*.csv"),
    "haiend_train_glob":       os.path.join(_DATA, "haiend-23.05","end-train*.csv"),
    "haiend_test_glob":        os.path.join(_DATA, "haiend-23.05","end-test*.csv"),
    "hai_label_train_glob":    os.path.join(_DATA, "hai-23.05",   "label-train*.csv"),
    "hai_label_test_glob":     os.path.join(_DATA, "hai-23.05",   "label-test*.csv"),
    "haiend_label_train_glob": os.path.join(_DATA, "haiend-23.05","label-train*.csv"),
    "haiend_label_test_glob":  os.path.join(_DATA, "haiend-23.05","label-test*.csv"),

    "timestamp_col": "timestamp",
    "attack_col":    "label",

    # ── Windows ───────────────────────────────────────────────────────────────
    "window_size": 60,
    "step_size":    1,

    # ── Model ─────────────────────────────────────────────────────────────────
    "hidden_size": 128,
    "num_layers":    2,
    "dropout":     0.2,
    "batch_size":  256,
    "epochs":       30,
    "lr":          1e-3,

    # ── Threshold ─────────────────────────────────────────────────────────────
    # "percentile" → use Nth percentile of train errors
    # "best_f1"    → sweep thresholds and pick best F1 on test set
    "threshold_strategy":    "percentile",
    "threshold_percentile":  99,

    # ── Output ────────────────────────────────────────────────────────────────
    "out_dir":        os.path.join(_REPO, "outputs", "bedour"),
    "model_save":     os.path.join(_REPO, "outputs", "bedour", "lstm_ae.pt"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {DEVICE}")
print(f"Repo   : {_REPO}")
print(f"Data   : {_DATA}")
print(f"Output : {CFG['out_dir']}\n")
os.makedirs(CFG["out_dir"], exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_glob(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    print(f"    {len(files)} file(s): {[os.path.basename(f) for f in files]}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_split(split):
    """Load + merge HAI, HAIEnd, and labels for 'train' or 'test'."""
    assert split in ("train", "test")
    ts  = CFG["timestamp_col"]
    atk = CFG["attack_col"]

    print(f"[{split.upper()}] HAI sensor data …")
    hai = load_glob(CFG[f"hai_{split}_glob"])
    hai.columns = hai.columns.str.strip()

    print(f"[{split.upper()}] HAIEnd sensor data …")
    haiend = load_glob(CFG[f"haiend_{split}_glob"])
    haiend.columns = haiend.columns.str.strip()
    haiend.rename(columns={"Timestamp": ts}, inplace=True)  # capital T fix

    # Drop label column from sensor files if accidentally present
    hai.drop(columns=[atk],    errors="ignore", inplace=True)
    haiend.drop(columns=[atk], errors="ignore", inplace=True)

    # Merge sensor data
    merged = pd.merge(hai, haiend, on=ts, suffixes=("_hai", "_haiend"))
    print(f"  Merged sensor shape: {merged.shape}")

    # Load attack labels from separate label files
    print(f"[{split.upper()}] Labels …")
    try:
        lbl_hai    = load_glob(CFG[f"hai_label_{split}_glob"])
        lbl_haiend = load_glob(CFG[f"haiend_label_{split}_glob"])
        lbl_hai.columns    = lbl_hai.columns.str.strip()
        lbl_haiend.columns = lbl_haiend.columns.str.strip()

        # Union: a timestamp is an attack if EITHER dataset labels it as one
        labels = pd.concat([lbl_hai, lbl_haiend], ignore_index=True)
        labels = labels.groupby(ts, as_index=False)[atk].max()
        merged = pd.merge(merged, labels, on=ts, how="left")
        merged[atk] = merged[atk].fillna(0).astype(int)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}\n  Setting all labels to 0.")
        merged[atk] = 0

    rate = merged[atk].mean() * 100
    print(f"  Final shape: {merged.shape}  |  attack rate: {rate:.2f}%\n")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(train_df, test_df):
    ts  = CFG["timestamp_col"]
    atk = CFG["attack_col"]

    feat_cols = [
        c for c in train_df.columns
        if c not in [ts, atk] and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    X_tr = train_df[feat_cols].values.astype(np.float32)
    X_te = test_df[feat_cols].values.astype(np.float32)
    y_te = test_df[atk].values.astype(int)

    # Domain-aware normalisation:
    # Binary-constant sensors (only 0/1, near-zero std) are left as-is
    # to avoid division-by-zero — this is the fix that gave +0.129 F1
    binary = np.array([
        np.std(X_tr[:, i]) < 0.01 and
        set(np.unique(X_tr[:, i])).issubset({0.0, 1.0})
        for i in range(X_tr.shape[1])
    ])
    print(f"Features total       : {len(feat_cols)}")
    print(f"Binary-const skipped : {binary.sum()}")
    print(f"Scaled normally      : {(~binary).sum()}\n")

    scaler  = MinMaxScaler()
    X_tr_s  = X_tr.copy()
    X_te_s  = X_te.copy()
    non_bin = np.where(~binary)[0]
    scaler.fit(X_tr[:, non_bin])
    X_tr_s[:, non_bin] = scaler.transform(X_tr[:, non_bin])
    X_te_s[:, non_bin] = scaler.transform(X_te[:, non_bin])

    X_tr_s = np.nan_to_num(X_tr_s, nan=0.0, posinf=1.0, neginf=0.0)
    X_te_s = np.nan_to_num(X_te_s, nan=0.0, posinf=1.0, neginf=0.0)

    return X_tr_s, X_te_s, y_te, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 3. SLIDING WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
def make_windows(data, window, step):
    idx = range(0, len(data) - window + 1, step)
    return np.stack([data[i:i+window] for i in idx])


def make_windows_labeled(data, labels, window, step):
    idx = range(0, len(data) - window + 1, step)
    X   = np.stack([data[i:i+window]         for i in idx])
    y   = np.array([labels[i:i+window].max() for i in idx])
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        d = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_size,  hidden_size, num_layers,
                               batch_first=True, dropout=d)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                               batch_first=True, dropout=d)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (h, c)  = self.encoder(x)
        dec_in     = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in, (h, c))
        return self.fc(dec_out)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(windows):
    n, seq, feat = windows.shape
    print(f"Training  —  windows: {n:,}  |  seq: {seq}  |  features: {feat}")

    t      = torch.tensor(windows, dtype=torch.float32)
    loader = DataLoader(TensorDataset(t, t),
                        batch_size=CFG["batch_size"], shuffle=True)

    model = LSTMAutoencoder(
        feat, CFG["hidden_size"], CFG["num_layers"], CFG["dropout"]
    ).to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    crit = nn.MSELoss()

    losses = []
    for ep in range(1, CFG["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        avg = total / n
        losses.append(avg)
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{CFG['epochs']}  loss: {avg:.6f}")

    torch.save(model.state_dict(), CFG["model_save"])
    print(f"\nModel saved → {CFG['model_save']}")
    return model, losses


# ─────────────────────────────────────────────────────────────────────────────
# 6. RECONSTRUCTION ERROR
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_errors(model, windows, batch=512):
    model.eval()
    out = []
    for s in range(0, len(windows), batch):
        xb    = torch.tensor(windows[s:s+batch], dtype=torch.float32).to(DEVICE)
        recon = model(xb)
        mse   = ((recon - xb) ** 2).mean(dim=(1, 2))
        out.extend(mse.cpu().numpy().tolist())
    return np.array(out)


# ─────────────────────────────────────────────────────────────────────────────
# 7. THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────
def get_threshold(train_err, test_err, y_test):
    if CFG["threshold_strategy"] == "percentile":
        thr = float(np.percentile(train_err, CFG["threshold_percentile"]))
        print(f"Threshold (p{CFG['threshold_percentile']} train errors): {thr:.6f}")
    else:
        best, thr = 0.0, 0.0
        for p in range(80, 100):
            t  = float(np.percentile(test_err, p))
            f1 = f1_score(y_test, (test_err > t).astype(int), zero_division=0)
            if f1 > best:
                best, thr = f1, t
        print(f"Best-F1 threshold: {thr:.6f}  (F1={best:.4f})")
    return thr


# ─────────────────────────────────────────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(errors, y_true, threshold):
    y_pred = (errors > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Threshold": round(float(threshold), 6),
        "F1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred),                   4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_true, errors),                    4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def print_metrics(m):
    print("\n" + "═"*52)
    print("  LSTM AUTOENCODER — RESULTS")
    print("═"*52)
    bar_keys = ["F1", "Accuracy", "Precision", "Recall", "ROC_AUC"]
    for k in bar_keys:
        v      = m[k]
        filled = int(round(v * 20))
        bar    = "[" + "█"*filled + "░"*(20-filled) + "]"
        print(f"  {k:<12}  {bar}  {v:.4f}")
    print(f"  {'Threshold':<12}: {m['Threshold']}")
    print(f"  {'TP':<12}: {m['TP']:,}   (attacks caught)")
    print(f"  {'TN':<12}: {m['TN']:,}   (normal passed)")
    print(f"  {'FP':<12}: {m['FP']:,}   (false alarms)")
    print(f"  {'FN':<12}: {m['FN']:,}   (missed attacks)")
    print("═"*52 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name):
    path = os.path.join(CFG["out_dir"], name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_training_loss(losses):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(losses)+1), losses, color=ACCENT, linewidth=2)
    ax.fill_between(range(1, len(losses)+1), losses, alpha=0.15, color=ACCENT)
    ax.set_title("Training Loss — LSTM Autoencoder", fontsize=14, pad=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_xlim(1, len(losses))
    fig.tight_layout()
    _save(fig, "01_training_loss.png")


def plot_confusion_matrix(m):
    tp, tn, fp, fn = m["TP"], m["TN"], m["FP"], m["FN"]
    cm    = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list("cm", ["#0F1117", "#1A3A5C", ACCENT])
    ax.imshow(cm, cmap=cmap, aspect="auto")

    lbl  = [["TN", "FP"], ["FN", "TP"]]
    cols = [[SUCCESS, DANGER], [DANGER, SUCCESS]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i-0.18, lbl[i][j],
                    ha="center", va="center", fontsize=15,
                    fontweight="bold", color=cols[i][j])
            ax.text(j, i+0.10, f"{cm[i,j]:,}",
                    ha="center", va="center", fontsize=13, color="#E8EAF0")
            ax.text(j, i+0.35, f"({100*cm[i,j]/total:.1f}%)",
                    ha="center", va="center", fontsize=10, color=MUTED)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Normal", "Predicted Attack"])
    ax.set_yticklabels(["Actual Normal",    "Actual Attack"])
    ax.set_title("Confusion Matrix", fontsize=14, pad=12)
    fig.tight_layout()
    _save(fig, "02_confusion_matrix.png")


def plot_roc(errors, y_true, auc):
    fpr, tpr, _ = roc_curve(y_true, errors)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2.5,
            label=f"LSTM Autoencoder  (AUC = {auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color=ACCENT)
    ax.plot([0,1],[0,1], "--", color=MUTED, linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, pad=12)
    ax.legend(loc="lower right", framealpha=0.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, "03_roc_curve.png")


def plot_precision_recall(errors, y_true):
    prec, rec, _ = precision_recall_curve(y_true, errors)
    ap = average_precision_score(y_true, errors)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color=WARNING, linewidth=2.5,
            label=f"LSTM AE  (AP = {ap:.4f})")
    ax.axhline(y_true.mean(), linestyle="--", color=MUTED, linewidth=1,
               label=f"Baseline (attack rate = {y_true.mean():.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, pad=12)
    ax.legend(loc="upper right", framealpha=0.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, "04_precision_recall.png")


def plot_error_dist(train_err, test_err, y_test, threshold):
    fig, ax = plt.subplots(figsize=(10, 5))
    kw = dict(bins=300, alpha=0.55, density=True, histtype="stepfilled")
    ax.hist(train_err,              color=MUTED,   label="Train (normal)", **kw)
    ax.hist(test_err[y_test == 0],  color=SUCCESS, label="Test – normal",  **kw)
    ax.hist(test_err[y_test == 1],  color=DANGER,  label="Test – attack",  **kw)
    ax.axvline(threshold, color=WARNING, linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("Reconstruction Error (MSE)"); ax.set_ylabel("Density (log)")
    ax.set_title("Reconstruction Error Distribution", fontsize=14, pad=12)
    ax.legend(framealpha=0.2)
    fig.tight_layout()
    _save(fig, "05_error_distribution.png")


def plot_metrics_table(m):
    rows = [
        ("F1 Score",   f"{m['F1']:.4f}",        "Primary anomaly detection score"),
        ("Accuracy",   f"{m['Accuracy']:.4f}",   "Overall correct predictions"),
        ("Precision",  f"{m['Precision']:.4f}",  "Of flagged attacks, how many were real"),
        ("Recall",     f"{m['Recall']:.4f}",     "Of real attacks, how many were caught"),
        ("ROC-AUC",    f"{m['ROC_AUC']:.4f}",    "Area under ROC curve"),
        ("Threshold",  f"{m['Threshold']:.6f}",  "Reconstruction error cut-off"),
        ("TP",         f"{m['TP']:,}",            "Attacks correctly detected  ✓"),
        ("TN",         f"{m['TN']:,}",            "Normal traffic correctly passed  ✓"),
        ("FP",         f"{m['FP']:,}",            "False alarms  ✗"),
        ("FN",         f"{m['FN']:,}",            "Missed attacks  ✗  ← keep low"),
    ]
    n     = len(rows)
    fig_h = 0.55 * n + 1.4
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")

    hx = [0.01, 0.20, 0.39]
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 1-1/(n+1)), 1, 1/(n+1),
        boxstyle="square,pad=0", transform=ax.transAxes,
        color="#1A3A5C", zorder=2))
    for x, h in zip(hx, ["Metric", "Value", "Description"]):
        ax.text(x, 1-0.5/(n+1), h, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=ACCENT, va="center")

    good   = {"F1 Score","ROC-AUC","TP","TN"}
    bad    = {"FP","FN"}
    bg_map = {**{k:"#1C3A28" for k in good}, **{k:"#3A1C1C" for k in bad}}

    for i, (metric, value, desc) in enumerate(rows):
        y_top = 1-(i+1)/(n+1)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y_top), 1, 1/(n+1),
            boxstyle="square,pad=0", transform=ax.transAxes,
            color=bg_map.get(metric, "#161B27"), zorder=1))
        y_mid = y_top + 0.5/(n+1)
        vc = SUCCESS if metric in good else DANGER if metric in bad else "#E8EAF0"
        ax.text(hx[0], y_mid, metric, transform=ax.transAxes,
                fontsize=10, color="#E8EAF0", va="center")
        ax.text(hx[1], y_mid, value,  transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=vc, va="center")
        ax.text(hx[2], y_mid, desc,   transform=ax.transAxes,
                fontsize=9,  color=MUTED, va="center")

    ax.set_title("LSTM Autoencoder — Full Evaluation Results",
                 fontsize=14, pad=12, loc="left")
    _save(fig, "06_metrics_table.png")


def plot_dashboard(m, losses, train_err, test_err, y_test, threshold):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("LSTM Autoencoder — ICS Anomaly Detection Dashboard",
                 fontsize=16, color="#E8EAF0", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(losses)+1), losses, color=ACCENT, linewidth=2)
    ax1.fill_between(range(1, len(losses)+1), losses, alpha=0.15, color=ACCENT)
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch")

    # Error distribution
    ax2 = fig.add_subplot(gs[0, 1])
    kw  = dict(bins=200, alpha=0.55, density=True, histtype="stepfilled")
    ax2.hist(train_err,             color=MUTED,   label="Train",         **kw)
    ax2.hist(test_err[y_test == 0], color=SUCCESS, label="Test – normal", **kw)
    ax2.hist(test_err[y_test == 1], color=DANGER,  label="Test – attack", **kw)
    ax2.axvline(threshold, color=WARNING, linewidth=1.5,
                linestyle="--", label="Threshold")
    ax2.set_yscale("log"); ax2.set_title("Error Distribution")
    ax2.legend(fontsize=8, framealpha=0.2)

    # ROC curve
    ax3 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_test, test_err)
    ax3.plot(fpr, tpr, color=ACCENT, linewidth=2,
             label=f"AUC = {m['ROC_AUC']:.4f}")
    ax3.plot([0,1],[0,1], "--", color=MUTED, linewidth=1)
    ax3.fill_between(fpr, tpr, alpha=0.10, color=ACCENT)
    ax3.set_title("ROC Curve")
    ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR")
    ax3.legend(fontsize=9, framealpha=0.2)

    # Confusion matrix
    ax4 = fig.add_subplot(gs[1, 0])
    cm  = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    cmap = LinearSegmentedColormap.from_list("cm", ["#0F1117","#1A3A5C", ACCENT])
    ax4.imshow(cm, cmap=cmap, aspect="auto")
    lbl  = [["TN","FP"],["FN","TP"]]
    cols = [[SUCCESS, DANGER],[DANGER, SUCCESS]]
    for i in range(2):
        for j in range(2):
            ax4.text(j, i-0.18, lbl[i][j], ha="center", va="center",
                     fontsize=13, fontweight="bold", color=cols[i][j])
            ax4.text(j, i+0.18, f"{cm[i,j]:,}", ha="center", va="center",
                     fontsize=11, color="#E8EAF0")
    ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
    ax4.set_xticklabels(["Pred Normal","Pred Attack"])
    ax4.set_yticklabels(["Actual Normal","Actual Attack"])
    ax4.set_title("Confusion Matrix")

    # Precision-Recall
    ax5 = fig.add_subplot(gs[1, 1])
    prec, rec, _ = precision_recall_curve(y_test, test_err)
    ap = average_precision_score(y_test, test_err)
    ax5.plot(rec, prec, color=WARNING, linewidth=2, label=f"AP = {ap:.4f}")
    ax5.axhline(y_test.mean(), linestyle="--", color=MUTED, linewidth=1)
    ax5.set_title("Precision-Recall")
    ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
    ax5.legend(fontsize=9, framealpha=0.2)

    # Metric bar chart
    ax6 = fig.add_subplot(gs[1, 2])
    bm  = ["F1","Accuracy","Precision","Recall","ROC_AUC"]
    bv  = [m[k] for k in bm]
    bc  = [ACCENT if v >= 0.75 else WARNING if v >= 0.5 else DANGER for v in bv]
    bars = ax6.barh(["F1","Accuracy","Precision","Recall","ROC-AUC"],
                    bv, color=bc, height=0.5)
    ax6.set_xlim(0, 1.15)
    for bar, val in zip(bars, bv):
        ax6.text(val+0.02, bar.get_y()+bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=10, color="#E8EAF0")
    ax6.set_title("Metric Summary")
    ax6.axvline(0.5, color=MUTED, linewidth=0.5, linestyle=":")

    _save(fig, "07_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(m):
    path = os.path.join(CFG["out_dir"], "metrics.csv")
    pd.DataFrame([m]).to_csv(path, index=False)
    print(f"  Metrics CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Load data
    train_df = load_split("train")
    test_df  = load_split("test")

    # 2. Preprocess
    print("Preprocessing …")
    X_tr, X_te, y_te_raw, _ = preprocess(train_df, test_df)

    # 3. Windows — train on NORMAL only (no attack leakage)
    W, S = CFG["window_size"], CFG["step_size"]
    atk  = CFG["attack_col"]
    normal_mask   = train_df[atk].values == 0
    train_windows = make_windows(X_tr[normal_mask], W, S)
    test_windows, y_test = make_windows_labeled(X_te, y_te_raw, W, S)

    print(f"Train windows (normal only) : {len(train_windows):,}")
    print(f"Test  windows               : {len(test_windows):,}")
    print(f"Attack windows in test      : {int(y_test.sum()):,}\n")

    # 4. Train
    model, losses = train(train_windows)

    # 5. Errors
    print("\nComputing reconstruction errors …")
    train_err = get_errors(model, train_windows)
    test_err  = get_errors(model, test_windows)

    # 6. Threshold + metrics
    threshold = get_threshold(train_err, test_err, y_test)
    metrics   = evaluate(test_err, y_test, threshold)
    print_metrics(metrics)

    # 7. Save + plot
    print("Saving outputs …")
    save_csv(metrics)
    plot_training_loss(losses)
    plot_confusion_matrix(metrics)
    plot_roc(test_err, y_test, metrics["ROC_AUC"])
    plot_precision_recall(test_err, y_test)
    plot_error_dist(train_err, test_err, y_test, threshold)
    plot_metrics_table(metrics)
    plot_dashboard(metrics, losses, train_err, test_err, y_test, threshold)

    print(f"\nAll outputs saved to → {CFG['out_dir']}")
    print("Done ✓\n")


if __name__ == "__main__":
    main()
