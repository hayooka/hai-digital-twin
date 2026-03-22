"""
Conditional DDPM — Attack Generator (primary)

Data:   attack windows from test1+2 where attack==1
        Conditioned on attack_type (FDI / Replay / DoS) and physics bounds
Input:  (window_len, 277) — same shape as Transformer input
Output: novel attack windows clipped to physics-valid sensor bounds

Eval:   TSTR — generate synthetic attacks → train XGBoost → test on real test1+2

TIP: Pass norm from twin() if Digital Twin was already run,
     to keep normalization consistent across all models.
"""
from __future__ import annotations

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # add project root to path
from utils.prep import generate

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEAT       = 277            # number of sensor features
# N_FEAT     = 86             # HAI-only mode (commented out)
WINDOW_LEN   = 240            # 60 enc + 180 dec — matches Transformer seq2seq split
N_ATTACK_TYPES = 3            # FDI=0, Replay=1, DoS=2

# DDPM schedule
T_STEPS      = 1000           # diffusion timesteps
BETA_START   = 1e-4
BETA_END     = 0.02

# Model hyperparameters
D_MODEL      = 256
N_HEADS      = 8
N_LAYERS     = 4
FFN_DIM      = 512
DROPOUT      = 0.1

# Training hyperparameters
EPOCHS       = 100
BATCH        = 32
LR           = 2e-4

Path("outputs").mkdir(exist_ok=True)


# ── DDPM noise schedule ────────────────────────────────────────────────────────

def make_beta_schedule(T: int, beta_start: float, beta_end: float) -> dict[str, torch.Tensor]:
    """
    Linear beta schedule with all derived quantities pre-computed.

    Returns dict with tensors needed for forward / reverse diffusion:
        betas, alphas, alpha_bars, sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars, posterior_variance
    """
    betas      = torch.linspace(beta_start, beta_end, T)           # (T,)
    alphas     = 1.0 - betas                                        # (T,)
    alpha_bars = torch.cumprod(alphas, dim=0)                       # (T,)

    alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)    # shift right

    return {
        "betas":                    betas,
        "alphas":                   alphas,
        "alpha_bars":               alpha_bars,
        "sqrt_alpha_bars":          alpha_bars.sqrt(),
        "sqrt_one_minus_alpha_bars": (1.0 - alpha_bars).sqrt(),
        "posterior_variance":       betas * (1.0 - alpha_bars_prev)
                                    / (1.0 - alpha_bars).clamp(min=1e-12),
    }


# ── Positional Encoding ────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — identical to transformer_model.py."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ── Timestep Embedding ─────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """
    Maps diffusion timestep t (scalar integer) to a dense embedding.
    Uses sinusoidal encoding followed by two linear projections.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long → (B, d_model)"""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device).float() / half
        )
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)
        return self.proj(emb)                                           # (B, d_model)


# ── Conditional Diffusion Model ────────────────────────────────────────────────

class ConditionalDiffusion(nn.Module):
    """
    Conditional DDPM denoising network for ICS attack window generation.

    Architecture
    ------------
    Input:  noisy window x_t  (B, window_len, n_features)
    Cond:   attack_type label (B,) integer  [FDI=0, Replay=1, DoS=2]
            physics_bounds    (B, 2, n_features)  — [min, max] per sensor
    Output: predicted noise epsilon (B, window_len, n_features)

    The model is a Transformer encoder that processes the sequence with
    additional conditioning injected via:
      - Timestep embedding added to every token
      - Attack-type embedding added to every token
      - Physics bounds projected to d_model and added as a prefix [CLS] token
    """

    def __init__(
        self,
        n_features:     int = N_FEAT,
        window_len:     int = WINDOW_LEN,
        n_attack_types: int = N_ATTACK_TYPES,
        d_model:        int = D_MODEL,
        n_heads:        int = N_HEADS,
        n_layers:       int = N_LAYERS,
        ffn_dim:        int = FFN_DIM,
        dropout:        float = DROPOUT,
        T:              int = T_STEPS,
        beta_start:     float = BETA_START,
        beta_end:       float = BETA_END,
    ):
        super().__init__()
        self.n_features     = n_features
        self.window_len     = window_len
        self.n_attack_types = n_attack_types
        self.d_model        = d_model
        self.T              = T

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj  = nn.Linear(n_features, d_model)
        self.output_proj = nn.Linear(d_model, n_features)
        self.pos_enc     = PositionalEncoding(d_model, max_len=window_len + 2,
                                              dropout=dropout)

        # ── Conditioning modules ──────────────────────────────────────────────
        # Timestep embedding
        self.time_emb = TimestepEmbedding(d_model)

        # Attack-type embedding (one additional "null" class for unconditional)
        self.attack_emb = nn.Embedding(n_attack_types + 1, d_model)

        # Physics bounds: flatten (2, n_features) → project to d_model
        # Used as a learnable prefix / CLS token for the sequence
        self.bounds_proj = nn.Sequential(
            nn.Linear(2 * n_features, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # ── Transformer encoder backbone ──────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, n_layers, enable_nested_tensor=False
        )

        # ── Noise schedule (registered as buffers — not parameters) ───────────
        schedule = make_beta_schedule(T, beta_start, beta_end)
        for name, val in schedule.items():
            self.register_buffer(name, val)

    # ── Conditioning helper ───────────────────────────────────────────────────

    def _build_cond(
        self,
        t:            torch.Tensor,    # (B,)
        attack_type:  torch.Tensor,    # (B,) long
        bounds_flat:  torch.Tensor,    # (B, 2 * n_features)
    ) -> torch.Tensor:
        """Returns (B, d_model) combined conditioning vector."""
        t_emb   = self.time_emb(t)                  # (B, d_model)
        a_emb   = self.attack_emb(attack_type)       # (B, d_model)
        b_emb   = self.bounds_proj(bounds_flat)      # (B, d_model)
        return t_emb + a_emb + b_emb                 # (B, d_model)

    # ── Forward: predict noise ─────────────────────────────────────────────────

    def forward(
        self,
        x_t:         torch.Tensor,   # (B, window_len, n_features)  noisy input
        t:           torch.Tensor,   # (B,)  diffusion timestep
        attack_type: torch.Tensor,   # (B,)  long: 0=FDI 1=Replay 2=DoS
        bounds:      torch.Tensor,   # (B, 2, n_features)  [min; max] normalized
    ) -> torch.Tensor:               # → (B, window_len, n_features)  predicted noise
        B = x_t.size(0)

        # Flatten bounds and compute conditioning vector
        bounds_flat = bounds.view(B, -1)                           # (B, 2*n_features)
        cond        = self._build_cond(t, attack_type, bounds_flat)  # (B, d_model)

        # Project input sequence
        h = self.input_proj(x_t)                                   # (B, T, d_model)

        # Inject conditioning into every token (broadcast over sequence dim)
        h = h + cond.unsqueeze(1)                                  # (B, T, d_model)

        # Prepend a physics/conditioning CLS token
        cls_token = cond.unsqueeze(1)                              # (B, 1, d_model)
        h = torch.cat([cls_token, h], dim=1)                       # (B, T+1, d_model)

        # Positional encoding + transformer
        h   = self.pos_enc(h)                                      # (B, T+1, d_model)
        h   = self.encoder(h)                                       # (B, T+1, d_model)

        # Drop CLS token, project back to sensor space
        h   = h[:, 1:, :]                                          # (B, T, d_model)
        eps = self.output_proj(h)                                   # (B, T, n_features)
        return eps

    # ── DDPM forward process ───────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,    # (B, T, F) clean sample
        t:  torch.Tensor,    # (B,) timestep indices
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0) = N(sqrt(alpha_bar_t)*x0, (1-alpha_bar_t)*I).
        Returns (x_t, epsilon).
        """
        if eps is None:
            eps = torch.randn_like(x0)
        sqrt_ab    = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab  = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * eps
        return x_t, eps

    # ── DDPM reverse process (sampling) ───────────────────────────────────────

    @torch.no_grad()
    def p_sample(
        self,
        x_t:         torch.Tensor,   # (B, T, F)
        t_idx:       int,
        attack_type: torch.Tensor,   # (B,) long
        bounds:      torch.Tensor,   # (B, 2, F)
    ) -> torch.Tensor:
        """Single reverse diffusion step: x_t → x_{t-1}."""
        B     = x_t.size(0)
        t_vec = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)

        beta        = self.betas[t_idx]
        alpha       = self.alphas[t_idx]
        alpha_bar   = self.alpha_bars[t_idx]
        sqrt_1mab   = self.sqrt_one_minus_alpha_bars[t_idx]

        # Predict noise
        eps_pred = self.forward(x_t, t_vec, attack_type, bounds)

        # Compute mean of p(x_{t-1} | x_t)
        coef     = beta / sqrt_1mab
        mean     = (1.0 / alpha.sqrt()) * (x_t - coef * eps_pred)

        if t_idx == 0:
            return mean

        # Add noise scaled by posterior variance
        post_var = self.posterior_variance[t_idx]
        noise    = torch.randn_like(x_t)
        return mean + post_var.sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        B:           int,
        attack_type: torch.Tensor,   # (B,) long
        bounds:      torch.Tensor,   # (B, 2, F)
        device:      torch.device,
        physics_clip: bool = True,
    ) -> torch.Tensor:
        """
        Full reverse diffusion: pure noise → clean attack window.

        Parameters
        ----------
        B            : batch size (number of windows to generate)
        attack_type  : (B,) integer labels 0=FDI 1=Replay 2=DoS
        bounds       : (B, 2, n_features)  normalized [min, max] per sensor
        device       : torch device
        physics_clip : if True, clip final output to [bounds[:,0], bounds[:,1]]

        Returns
        -------
        x0 : (B, window_len, n_features)  generated attack windows
        """
        self.eval()
        x = torch.randn(B, self.window_len, self.n_features, device=device)

        for t_idx in reversed(range(self.T)):
            x = self.p_sample(x, t_idx, attack_type, bounds)

        if physics_clip:
            lo = bounds[:, 0, :].unsqueeze(1)   # (B, 1, F)
            hi = bounds[:, 1, :].unsqueeze(1)   # (B, 1, F)
            x  = torch.clamp(x, lo, hi)

        return x


# ── Physics bounds helper ──────────────────────────────────────────────────────

def _extract_physics_bounds(norm) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-sensor [min, max] bounds from a fitted HAISensorNormalizer.

    For a zscore normalizer the bounds in normalized space are derived from
    the per-column stats.  We define physics bounds as ±4σ in the original
    scale mapped back to normalized space → simply [-4, +4] per sensor.

    For a minmax normalizer, the normalized range is always [0, 1].

    Returns arrays of shape (n_features,) in the normalized (model) space.
    """
    if norm.method == "zscore":
        # In z-score space every sensor is N(0,1) by construction;
        # physics bounds are the empirical extremes captured in training.
        # We use ±4 as a conservative physics envelope.
        n = len(norm.sensor_cols)
        lo = np.full(n, -4.0, dtype=np.float32)
        hi = np.full(n,  4.0, dtype=np.float32)
    else:  # minmax
        n  = len(norm.sensor_cols)
        lo = np.zeros(n, dtype=np.float32)
        hi = np.ones(n,  dtype=np.float32)
    return lo, hi


# ── Training function ──────────────────────────────────────────────────────────

def train_diffusion(
    gen_data:   dict,
    norm,
    window_len:  int   = WINDOW_LEN,
    epochs:      int   = EPOCHS,
    batch:       int   = BATCH,
    lr:          float = LR,
) -> ConditionalDiffusion:
    """
    Train ConditionalDiffusion on test1 attack windows from generate().

    Parameters
    ----------
    gen_data   : dict returned by generate() — contains attack_windows.
    norm       : fitted HAISensorNormalizer — used to derive physics bounds.
    window_len : length of each window (240 = 60 enc + 180 dec).
    epochs     : number of training epochs.
    batch      : mini-batch size.
    lr         : Adam learning rate.

    Returns
    -------
    Trained ConditionalDiffusion model (on best training loss).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Diffusion] Using device: {device}")

    # ── Extract attack windows from generate() ────────────────────────────────
    attack_windows = gen_data["attack_windows"]   # (N_atk, window_len, 277)
    print(f"[Diffusion] Attack windows: {attack_windows.shape}")

    if len(attack_windows) == 0:
        raise RuntimeError("No attack windows found in test_data. "
                           "Check y_test_labels contains 1s.")

    # ── Physics bounds ────────────────────────────────────────────────────────
    lo_np, hi_np = _extract_physics_bounds(norm)
    lo_t = torch.tensor(lo_np, dtype=torch.float32)   # (277,)
    hi_t = torch.tensor(hi_np, dtype=torch.float32)   # (277,)

    # ── Attack-type labels ────────────────────────────────────────────────────
    # The HAI dataset does not expose per-window attack types in twin().
    # We assign attack types by sequence order within the attack windows:
    #   first third  → FDI=0, second third → Replay=1, last third → DoS=2
    # This is a best-effort heuristic; replace with real labels when available.
    N_atk = len(attack_windows)
    thirds = N_atk // 3
    atk_labels = np.zeros(N_atk, dtype=np.int64)
    atk_labels[thirds    : 2 * thirds] = 1   # Replay
    atk_labels[2 * thirds:]            = 2   # DoS
    atk_labels_t = torch.tensor(atk_labels, dtype=torch.long)

    # ── Build model ───────────────────────────────────────────────────────────
    model = ConditionalDiffusion(
        n_features=N_FEAT, window_len=window_len,
        n_attack_types=N_ATTACK_TYPES, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        dropout=DROPOUT, T=T_STEPS, beta_start=BETA_START, beta_end=BETA_END,
    ).to(device)
    print(f"[Diffusion] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    # ── Augmentation: jitter + scaling ───────────────────────────────────────
    # 46 original windows → augment to ~4x more training samples.
    # Jitter: add small Gaussian noise to each window.
    # Scaling: multiply each window by a random factor near 1.0.
    # Both preserve the attack structure while increasing data diversity.
    rng = np.random.default_rng(42)
    aug_copies = []
    aug_labels = []
    for _ in range(3):   # 3 augmented copies → 4x total
        # Jitter: σ = 1% of each sensor's std across the attack windows
        sigma = attack_windows.std(axis=(0, 1), keepdims=True) * 0.01
        jittered = attack_windows + rng.normal(0, sigma, attack_windows.shape)
        # Scaling: random factor in [0.95, 1.05] per window
        scale = rng.uniform(0.95, 1.05, (len(attack_windows), 1, 1))
        augmented = (jittered * scale).astype(np.float32)
        aug_copies.append(augmented)
        aug_labels.append(atk_labels.copy())

    attack_aug = np.concatenate([attack_windows] + aug_copies, axis=0)
    atk_labels_aug = np.concatenate([atk_labels] + aug_labels, axis=0)
    print(f"[Diffusion] After augmentation: {attack_aug.shape[0]} windows ({len(attack_windows)} original + {len(attack_aug)-len(attack_windows)} augmented)")

    X_tensor       = torch.tensor(attack_aug,    dtype=torch.float32)
    atk_labels_t   = torch.tensor(atk_labels_aug, dtype=torch.long)
    N         = len(X_tensor)
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        idx   = np.random.permutation(N)
        total = 0.0
        n_batches = 0

        for i in range(0, N, batch):
            b   = idx[i : i + batch]
            x0  = X_tensor[b].to(device)                            # (B, T, F)
            B_  = x0.size(0)

            # Conditioning: attack type
            atk = atk_labels_t[b].to(device)                        # (B,)

            # Conditioning: physics bounds — same for every sample in batch
            lo_b = lo_t.unsqueeze(0).expand(B_, -1).to(device)      # (B, F)
            hi_b = hi_t.unsqueeze(0).expand(B_, -1).to(device)      # (B, F)
            bounds = torch.stack([lo_b, hi_b], dim=1)                # (B, 2, F)

            # Sample random timestep for each sample
            t_rand = torch.randint(0, model.T, (B_,), device=device)

            # Forward diffusion (add noise)
            x_t, eps = model.q_sample(x0, t_rand)

            # Predict noise
            eps_pred = model(x_t, t_rand, atk, bounds)

            # Simple noise-prediction MSE loss
            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total     += loss.item()
            n_batches += 1

        avg_loss = total / max(1, n_batches)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"[Diffusion] Best training loss: {best_loss:.6f}")
    return model


# ── Generation function ────────────────────────────────────────────────────────

def generate_attacks(
    model:       ConditionalDiffusion,
    attack_type: int | list[int],
    n_samples:   int,
    norm,
    physics_clip: bool = True,
) -> torch.Tensor:
    """
    Generate novel attack windows using the trained diffusion model.

    Parameters
    ----------
    model        : trained ConditionalDiffusion.
    attack_type  : integer label(s) for attack class.
                   0=FDI, 1=Replay, 2=DoS.
                   Can be a single int (all samples same type) or list of ints
                   of length n_samples for mixed-type generation.
    n_samples    : number of windows to generate.
    norm         : fitted HAISensorNormalizer — for deriving physics bounds.
    physics_clip : if True, clip output to physics-valid bounds (default True).

    Returns
    -------
    generated : (n_samples, window_len, 277) tensor on CPU.
    """
    device = next(model.parameters()).device
    model.eval()

    # ── Build attack_type tensor ──────────────────────────────────────────────
    if isinstance(attack_type, int):
        atk_t = torch.full((n_samples,), attack_type,
                           dtype=torch.long, device=device)
    else:
        atk_arr = np.array(attack_type, dtype=np.int64)
        if len(atk_arr) != n_samples:
            raise ValueError(
                f"attack_type list length ({len(atk_arr)}) "
                f"must match n_samples ({n_samples})"
            )
        atk_t = torch.tensor(atk_arr, dtype=torch.long, device=device)

    # ── Physics bounds ────────────────────────────────────────────────────────
    lo_np, hi_np = _extract_physics_bounds(norm)
    lo_t  = torch.tensor(lo_np, dtype=torch.float32, device=device)
    hi_t  = torch.tensor(hi_np, dtype=torch.float32, device=device)
    lo_b  = lo_t.unsqueeze(0).expand(n_samples, -1)   # (n_samples, F)
    hi_b  = hi_t.unsqueeze(0).expand(n_samples, -1)   # (n_samples, F)
    bounds = torch.stack([lo_b, hi_b], dim=1)          # (n_samples, 2, F)

    # ── Run reverse diffusion in mini-batches to avoid OOM ───────────────────
    chunk      = 64
    all_chunks = []
    for start in range(0, n_samples, chunk):
        end   = min(start + chunk, n_samples)
        atk_c = atk_t[start:end]
        bnd_c = bounds[start:end]
        x_c   = model.sample(end - start, atk_c, bnd_c, device,
                              physics_clip=physics_clip)
        all_chunks.append(x_c.cpu())

    generated = torch.cat(all_chunks, dim=0)   # (n_samples, window_len, F)
    print(f"[Diffusion] Generated {generated.shape[0]} attack windows "
          f"of shape {tuple(generated.shape[1:])}")
    return generated


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    # generate() loads test1 attack windows (240 steps = 60 enc + 180 dec).
    # test2 is held-out and never touched here.
    print("\nLoading attack windows via generate()...")
    data = generate(window_len=WINDOW_LEN, stride=60)
    norm = data["norm"]

    # ── 2. Train diffusion model ──────────────────────────────────────────────
    print("\nTraining Conditional Diffusion Model...")
    model = train_diffusion(data, norm)
    model.to(device)

    # ── 3. Generate novel attack windows (one batch per attack type) ──────────
    ATTACK_NAMES = {0: "FDI", 1: "Replay", 2: "DoS"}
    N_PER_TYPE   = 50   # generate 50 windows per attack type

    all_generated  = []
    all_type_labels = []

    for atk_id, atk_name in ATTACK_NAMES.items():
        print(f"\nGenerating {N_PER_TYPE} {atk_name} attack windows...")
        gen = generate_attacks(model, atk_id, N_PER_TYPE, norm, physics_clip=True)
        all_generated.append(gen)
        all_type_labels.extend([atk_id] * N_PER_TYPE)

    generated_windows = torch.cat(all_generated, dim=0)   # (150, window_len, 277)
    attack_type_labels = torch.tensor(all_type_labels, dtype=torch.long)

    print(f"\nTotal generated windows: {generated_windows.shape}")
    print(f"  Each window: {WINDOW_LEN} steps (first 60=enc input, last 180=dec target)")

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    save_path = Path("outputs/diffusion_attacks.pt")
    torch.save(
        {
            "generated_windows":  generated_windows,    # (150, window_len, 277)
            "attack_type_labels": attack_type_labels,   # (150,) 0=FDI 1=Replay 2=DoS
            "model_state":        model.state_dict(),
            "hyperparams": {
                "window_len":     WINDOW_LEN,
                "n_feat":         N_FEAT,
                "d_model":        D_MODEL,
                "n_heads":        N_HEADS,
                "n_layers":       N_LAYERS,
                "T_steps":        T_STEPS,
                "epochs":         EPOCHS,
            },
        },
        save_path,
    )
    print(f"\nSaved:")
    print(f"  {save_path}   (generated windows + model state)")
    print(f"    generated_windows  {tuple(generated_windows.shape)}")
    print(f"    attack_type_labels {tuple(attack_type_labels.shape)}")
