"""
run.py — Full pipeline: Digital Twin → Anomaly Detection → Attack Generator

Order matters:
    1. Digital Twin (Transformer + LSTM) — train on benign, compute errors
    2. Anomaly Detection (ISO Forest + LSTM-AE) — use those errors / windows
    3. Attack Generator (Diffusion + VAE + LSTM-AE) — generate synthetic attacks

Run:
    python run.py
"""

from utils.prep import twin, detect_errors, detect, generate

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DIGITAL TWIN
# Train on benign data. After training, compute reconstruction errors on
# train4 (val) and test1+2 — those errors feed the Anomaly Detection step.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 1: DIGITAL TWIN")
print("="*60)

data_twin = twin(input_len=60, target_len=180, stride=1)

X_train = data_twin["X_train"]   # (N, 60, 277)  benign train1+2+3
Y_train = data_twin["Y_train"]   # (N, 180, 277)
X_val   = data_twin["X_val"]     # (M, 60, 277)  benign train4
Y_val   = data_twin["Y_val"]     # (M, 180, 277)
X_test  = data_twin["X_test"]    # (K, 60, 277)  test1+2 all rows
Y_test  = data_twin["Y_test"]    # (K, 180, 277)
y_test  = data_twin["y_test_labels"]   # (K,) attack labels
norm    = data_twin["norm"]            # save for Attack Generator

# ── 1a. Transformer (Seq2Seq) ─────────────────────────────────────────────────
from models.transformer_model import TransformerSeq2Seq

transformer = TransformerSeq2Seq()
transformer.fit(X_train, Y_train, X_val, Y_val)

# Compute per-sensor reconstruction errors (needed for ISO Forest)
transformer_train_errors = transformer.reconstruction_errors(X_val)    # (M, 277)
transformer_test_errors  = transformer.reconstruction_errors(X_test)   # (K, 277)

# ── 1b. LSTM Seq2Seq ──────────────────────────────────────────────────────────
from models.lstm_model import LSTMSeq2Seq

lstm_twin = LSTMSeq2Seq()
lstm_twin.fit(X_train, Y_train, X_val, Y_val)

lstm_train_errors = lstm_twin.reconstruction_errors(X_val)    # (M, 277)
lstm_test_errors  = lstm_twin.reconstruction_errors(X_test)   # (K, 277)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ANOMALY DETECTION
# Uses reconstruction errors from the Digital Twin (not raw sensor data).
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 2: ANOMALY DETECTION")
print("="*60)

# ── 2a. ISO Forest — on Transformer errors ────────────────────────────────────
from models.iso_forest import HAIIsoForest

data_detect = detect_errors(
    train_errors = transformer_train_errors,   # (M, 277)
    test_errors  = transformer_test_errors,    # (K, 277)
    y_test       = y_test,
)

iso = HAIIsoForest()
iso.fit(data_detect["X_train"])
iso_preds = iso.predict(data_detect["X_test"])   # -1=anomaly, 1=normal

# ── 2b. LSTM Autoencoder (PD2 baseline) — on raw windows ─────────────────────
from models.lstm_autoencoder import LSTMAutoencoder

data_ae = detect(window_size=60, stride=1)

lstm_ae = LSTMAutoencoder()
lstm_ae.fit(data_ae["X_train"], data_ae["X_val"])   # input == target
lstm_ae_errors = lstm_ae.reconstruction_errors(data_ae["X_test"])   # (K, 277)
lstm_ae_preds  = lstm_ae.predict(data_ae["X_test"])


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ATTACK GENERATOR
# Trains on real attack rows from test1+2.
# Evaluated with TSTR: generate synthetic → train XGBoost → test on real data.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 3: ATTACK GENERATOR")
print("="*60)

data_gen = generate(norm=norm)   # pass norm from Digital Twin

attack_rows  = data_gen["attack_rows"]   # (N, 277) real attacks to learn from
normal_rows  = data_gen["normal_rows"]   # (M, 277) normal rows for conditioning
X_test_tstr  = data_gen["X_test_all"]   # (K, 277) for TSTR evaluation
y_test_tstr  = data_gen["y_test_all"]   # (K,) labels

# ── 3a. Diffusion Model ───────────────────────────────────────────────────────
from models.diffusion_model import DiffusionModel

diffusion = DiffusionModel()
diffusion.fit(attack_rows, normal_rows)         # class-conditioned
synthetic_diffusion = diffusion.generate(n=len(attack_rows))   # (N, 277)

# ── 3b. VAE ───────────────────────────────────────────────────────────────────
from models.vae_model import VAE

vae = VAE()
vae.fit(attack_rows, normal_rows)
synthetic_vae = vae.generate(n=len(attack_rows))   # (N, 277)

# ── 3c. LSTM Autoencoder (generator from PD2) ────────────────────────────────
from models.lstm_autoencoder import LSTMGenerativeAE

lstm_gen = LSTMGenerativeAE()
lstm_gen.fit(attack_rows)
synthetic_lstm = lstm_gen.generate(n=len(attack_rows))   # (N, 277)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

from evaluate import (
    eval_anomaly_detection,
    eval_tstr,
)

print("\n── Anomaly Detection ──")
eval_anomaly_detection("ISO Forest",   iso_preds,      data_detect["y_test"])
eval_anomaly_detection("LSTM-AE (PD2)", lstm_ae_preds, data_ae["y_test_labels"])

print("\n── Attack Generator (TSTR) ──")
eval_tstr("Diffusion", synthetic_diffusion, normal_rows, X_test_tstr, y_test_tstr)
eval_tstr("VAE",       synthetic_vae,       normal_rows, X_test_tstr, y_test_tstr)
eval_tstr("LSTM-AE",   synthetic_lstm,      normal_rows, X_test_tstr, y_test_tstr)
