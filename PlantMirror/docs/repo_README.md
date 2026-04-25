# HAI Digital Twin

A closed-loop digital twin of the HAI 23.05 industrial control system, trained to simulate normal and attack-injected plant behaviour across multiple control loops.

---

## Team Workflow

### Daily workflow
```bash
git pull                         # before starting
git add .
git commit -m "describe your work"
git push                         # after finishing
```

### Branching
```bash
git checkout -b your-feature-name
git push origin your-feature-name
```

**Rules:** always `git pull` before starting — do not edit other people's files.

---

## Project Structure

```
hai-digital-twin/
├── 02_data_pipeline/
│   ├── config.py              # loop definitions, column names, hyperparams
│   ├── scaled_split.py        # raw CSV → normalised sliding windows (.npz)
│   └── pipeline.py            # load windows, split into plant + controller arrays
│
├── 03_model/
│   ├── gru.py                 # GRUPlant, GRUController, CCClassifierRegressor
│   ├── lstm.py                # LSTMPlant (same interface as GRUPlant)
│   ├── transformer.py         # TransformerPlant (same interface)
│   ├── train_gru.py           # train GRU plant + all controllers
│   ├── train_lstm.py          # train LSTM plant + all controllers
│   ├── train_transformer.py   # train Transformer plant + all controllers
│   └── plot_results.py        # shared plotting utilities (loss curves + horizon plots)
│
├── outputs/
│   ├── scaled_split/          # preprocessed windows (already generated)
│   ├── gru_plant/             # GRU checkpoints + plots
│   ├── lstm_plant/            # LSTM checkpoints + plots
│   └── transformer_plant/     # Transformer checkpoints + plots
│
└── boiler_twin/               # separate boiler system experiments
```

---

## Architecture

### Closed-loop system

```
[SP, PV history]  →  GRUController × 4 (PC, LC, FC, TC)  →  CV sequence
[TIT03, PP04SP]   →  CCClassifierRegressor                →  pump on/off + speed
                                        │
                              Plant model (choice below)
                                        │
                    Encoder: non-PV signals + scenario embedding  →  hidden h
                    Decoder: step-by-step autoregressive rollout
                             input_t = [ cv_target_t  ‖  pv_{t-1} ]
                             pv_t   = FC( RNN/Transformer(input_t, h) )
                                        │
                                        ▼
                          5 PV outputs: P1_PIT01, P1_LIT01, P1_FT03Z,
                                        P1_TIT01, P1_TIT03
```

### Plant backbones

| Script | Plant model | Config |
|---|---|---|
| `train_gru.py` | GRU encoder + GRU decoder | hidden=256, layers=2 |
| `train_lstm.py` | LSTM encoder + LSTM decoder | hidden=256, layers=2 |
| `train_transformer.py` | Transformer encoder | d_model=128, heads=8, layers=3 |

Controllers are always `GRUController` regardless of plant backbone.

### Training order per epoch

1. `GRUController × 4` (PC, LC, FC, TC) — MSE on CV sequence
2. `CCClassifierRegressor` — BCE (pump on/off) + MSE (pump speed)
3. Plant model — scheduled-sampling MSE on PV
4. Validation — open-loop (ss_ratio=0), drives LR scheduler

Scheduled sampling ramps from 0 → 0.5 between epochs 10–100.

---

## Data

**HAI 23.05** — 1 Hz industrial control system logs with labelled attack segments.

| Split | Source |
|---|---|
| Train | train1 (100%) + train2 (100%) + train3 (first 30%) |
| Val | train3 (last 70%) |
| Test | train4 (fully held out — never seen during training) |

Attack types: `AP_no_combination`, `AP_with_combination`, `AE_no_combination`.

Input window: 300 steps (5 min) → Target window: 180 steps (3 min), stride: 60 steps.

---

## Running

Data preprocessing is already done. Just run the model you want to try:

```bash
python 03_model/train_gru.py
python 03_model/train_lstm.py
python 03_model/train_transformer.py
```

Each run saves to its `outputs/` subdirectory:

```
*_loss_curves.png        # training loss (log scale) + scheduled-sampling ratio
*_horizon_300s.png       # 5-min closed-loop rollout vs ground truth (5 PVs)
*_horizon_600s.png       # 10-min
*_horizon_900s.png       # 15-min
*_horizon_1800s.png      # 30-min
*.pt                     # model checkpoints
```
