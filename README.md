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
