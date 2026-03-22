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

| Metric | Value |
|--------|-------|
| RMSE normal | 0.097 |
| RMSE known attacks | 6.635 |
| Attack/Normal ratio | 68x |
| RMSE novel attacks | 4.028 |
| Generalization Gap | 2.607 |
| Acceptance rate | 100% |
| ISO Forest F1 | 0.346 |
| ISO Forest ROC-AUC | 0.659 |
