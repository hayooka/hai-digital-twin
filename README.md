# HAI Digital Twin Project

## 👥 Team Workflow

We use GitHub to collaborate on this project.

### 🔁 Daily Workflow
Before starting work:
git pull

After finishing work:
git add .
git commit -m "describe your work"
git push


### 🌿 Branching (Recommended)
Each member works on their own branch:

git checkout -b your-feature-name
git push origin your-feature-name


---

## 📁 Project Structure

- `models/` → All models (LSTM, Transformer, VAE, Diffusion, ISO Forest)
- `utils/` → Data preprocessing (normalize, windows, graph)
- `twin/` → Digital Twin pipeline and logic
- `data/` → Dataset (not uploaded to GitHub)
- `evaluate.py` → Final evaluation
- `outputs/` → Saved models, metrics, and plots
  - `outputs/lstm/` → LSTM results (causal + no_causal)
  - `outputs/transformer/` → Transformer results (causal + no_causal)
  - `outputs/causal_graph/` → Causal graph edges, parents, summary
  - `outputs/scaled_split/` → Preprocessed train/val/test splits
  - `outputs/RESULTS.md` → Full results summary

---

## ⚠️ Rules
- Always `git pull` before starting
- Do not edit other people's files
- Keep commits clear and simple

---

## 🧠 Idea
- Predict future system behavior (Digital Twin)
- Detect anomalies (ISO Forest)
- Generate attacks (Diffusion / VAE)

---

## 🏗️ Architecture

> [View full interactive diagram](docs/architecture.html)

**3 layers + Guided Generation (main research contribution):**

| Layer | Primary | Baseline | Data |
|-------|---------|----------|------|
| Physical Model | Transformer Seq2Seq | LSTM | train1-3 normal only |
| Detector | ISO Forest | LSTM Autoencoder | Transformer errors |
| Attack Generator | Conditional Diffusion | VAE | test1 attacks only |
| Guided Generation | Rejection Sampling | — | inference only |
| Final Evaluation | — | — | test2 blind test |

---

## 📊 Results

> Full results and analysis: [outputs/RESULTS.md](outputs/RESULTS.md)

### Model Comparison

| Model | Variant | RMSE Val | RMSE Normal | RMSE Attack | Separation Ratio | Causal Violation % |
|-------|---------|----------|-------------|-------------|-----------------|-------------------|
| LSTM | No Causal | 0.172 | 0.200 | 0.804 | 4.02 | 96.3% |
| LSTM | Causal | 0.260 | 0.284 | 1.214 | **4.28** | **13.1%** |
| Transformer | No Causal | **0.137** | **0.145** | 0.307 | 2.12 | 96.6% |
| Transformer | Causal | 0.238 | 0.251 | 0.923 | 3.68 | **13.0%** |

- **Causal constraints** reduce causal violations from ~96% → ~13% for both architectures
- **LSTM + Causal** achieves the best attack/normal separation ratio (4.28×)
- **Transformer + No Causal** achieves the lowest reconstruction error on normal data
