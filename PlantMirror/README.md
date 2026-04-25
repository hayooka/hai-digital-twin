# PlantMirror — Final Graduation Project Submission

**Project:** A digital twin of the HAI 23.05 P1 industrial boiler with attack simulation, classifier validation, and a multi-tab interactive dashboard.

**Track:** Industrial Control System (ICS) Security · Digital Twin Modelling

**Department:** Electrical Engineering

**Year:** 2026

**Supervisor:** Dr. Khaled Chahine

**Team:**

| Name | Discipline |
|---|---|
| Farah AlHayek | Computer Engineering |
| Mudhi Aldihani | Electrical Engineering |
| Bedour Mahdi | Computer Engineering |
| Fatma Abdulaziz | Electrical Engineering |
| Mariam Alsahhaf | Computer Engineering |

---

## Live deployment

| Component | Hosted on | URL |
|---|---|---|
| **Public website** | Netlify | https://resonant-cobbler-711d31.netlify.app |
| **Streamlit dashboard** | Hugging Face Spaces | https://MND4-plantmirror.hf.space |
| **Front-end source** | GitHub | https://github.com/Modhi2o2/PLANT_MIRROR_DASHBOARD |

Both URLs are permanent and run 24/7 without local hardware.

---

## What's in this archive

```
PlantMirror/
├── README.md                    ← THIS FILE
│
├── frontend/                    Next.js website (deployed on Netlify)
│   ├── app/                     route + layout
│   ├── components/              Hero, Plant, Team, Reviews, Navigation, Footer
│   ├── lib/                     Supabase client (lazy-init proxy)
│   ├── hooks/                   custom React hooks
│   ├── public/                  static assets
│   ├── supabase/                schema notes
│   ├── package.json             npm dependencies
│   ├── netlify.toml             Netlify build config
│   ├── next.config.js           Next.js runtime config
│   ├── tailwind.config.ts       theme + colors
│   └── tsconfig.json            TypeScript config
│
├── backend/                     Streamlit dashboard (deployed on HF Spaces)
│   ├── dashboard/               4-tab Streamlit app + chatbot sidebar
│   │   ├── app.py               entry point
│   │   ├── data_loader.py       cached loaders (with @st.cache_resource)
│   │   ├── components.py        plotly helpers + DARK/LIGHT CSS
│   │   ├── chatbot_sidebar.py   GPT-4o-mini sidebar
│   │   ├── tabs/                attack_sim, rollout, loop_explorer, classifier
│   │   ├── cache/               precomputed JSON + NPZ artifacts
│   │   ├── precompute_gallery.py
│   │   ├── run_classifier_experiments.py     XGBoost A/B/C
│   │   ├── run_classifier_experiments_rf.py  RF A/B/C/D
│   │   └── run_long_horizon_eval.py          1800-s NRMSE
│   ├── generator/               trained plant + 5 controllers + scalers
│   ├── attack_sim/              attack injection engine
│   ├── app/                     chatbot SYSTEM_PROMPT + assistive reasoner
│   ├── 01_causal_graph/         graph build script (TDMI lag estimation)
│   ├── 02_data_pipeline/        windowing + scaler-fit utilities
│   ├── 03_model/                plant training scripts
│   ├── 04_evaluate/             NRMSE per scenario / per PV
│   ├── 05_classifier/           IsoForest / OCSVM / AE baselines
│   ├── training/                long-horizon plant fine-tune (v2 ckpt)
│   ├── outputs/                 causal_graph + classifier bench JSON
│   ├── requirements.txt         pip dependencies
│   └── environment.yml          conda env
│
├── external_classifier/         XGBoost Hybrid Guardian (production detector)
│   ├── best_hai_classifier.pkl  the trained pipeline (scaler + model + features)
│   ├── model_builder.py         training script with Optuna 15-trial search
│   ├── generate_synthetic_dataset.py
│   ├── synthetic_attacks.csv    36k synthetic attack rows from the twin
│   └── inspect_classifier.py
│
├── data/                        HAI 23.05 P1 boiler test partitions
│   ├── test1.csv                79 MB (replay source for Tab 1 + Guardian eval)
│   └── test2.csv                336 MB (full eval set)
│
├── docs/                        Reference documentation
│   ├── TWIN_ARCHITECTURE.md     the 9-layer architecture diagram
│   ├── MODELS_HANDOFF.md        every model artifact, training, location
│   ├── CLASSIFIERS_REFERENCE.md XGBoost vs RF vs IsoForest comparison
│   ├── TWIN_BUILD_DIAGRAM.md    visual training pipeline
│   ├── repo_README.md           original repo README
│   └── PLAN.md                  the 5-hour build plan
│
└── audit_scripts/               QA scripts that verified accuracy pre-deploy
    ├── full_audit.py            10-phase audit, 147 of 149 pass
    ├── pre_deploy_audit.py      targeted XGBoost-swap checks, 79 of 79 pass
    ├── final_accuracy_audit.py  number-by-number traceability, 91 of 92 pass
    └── legacy_final_qa.py       earlier reference QA, 67 of 67 pass
```

---

## Headline numbers (every claim traceable to a source file)

| Claim | Number | Source artifact |
|---|---|---|
| Training data | 194 hours, normal-only | `train1+2+3.csv` row counts |
| Plant size | 5.25 M parameters | `backend/generator/weights/gru_plant.pt` |
| Forecast NRMSE at 180 s | 0.0095 | `backend/generator/weights/eval_results.json :: mean_nrmse` |
| Forecast NRMSE at 1800 s, Normal | 0.66 % | `backend/dashboard/cache/eval_1800s.json` |
| Forecast NRMSE at 1800 s, AP_with | 3.42 % | same |
| Classifier F1 | **0.587** | `external_classifier/best_hai_classifier.pkl` evaluated on real held-out test |
| Classifier AUROC | **0.904** | same |
| Augmentation gain (D − A) | **+0.327** | `backend/dashboard/cache/classifier_experiments.json` (A) + Guardian (D) |
| KS fidelity per loop (PASS/FAIL) | PC ✅ FC ✅ LC ❌ TC ❌ CC ❌ | Student Guide §3.3 |

**The headline scientific finding:**

> *Augmenting real attack training data with synthetic attack data generated by the digital twin lifts the XGBoost classifier's F1 from 0.260 to 0.587 — a +0.327 absolute improvement.*

---

## How to run locally

### Front-end (Next.js)

```bash
cd frontend
npm install
# Set environment variables (.env.local) — see frontend/README.md
npm run dev
# Opens http://localhost:3000
```

### Back-end (Streamlit dashboard)

```bash
cd backend
pip install -r requirements.txt
# Update absolute paths in dashboard/data_loader.py if needed
streamlit run dashboard/app.py --server.port 8504
# Opens http://localhost:8504
```

### Reproduce the classifier matrix

```bash
cd backend/dashboard
python run_classifier_experiments.py     # XGBoost A/B/C → cache/classifier_experiments.json
python run_classifier_experiments_rf.py  # Random Forest A/B/C/D → cache/classifier_experiments_rf.json
python run_long_horizon_eval.py --all    # 1800-s NRMSE → cache/eval_1800s.json
```

### Re-train the XGBoost Guardian

```bash
cd external_classifier
python model_builder.py
# Reads test1+test2 + synthetic_attacks.csv, writes best_hai_classifier.pkl
# Takes ~10 minutes on a modern CPU
```

---

## Architecture (9 layers)

```
Layer 9 — GPT-4o-mini chatbot sidebar (LiveState-grounded)
Layer 8 — Assistive reasoner (causal-graph BFS upstream)
Layer 7 — Causal graph (parents_full.json, 23 sensors, 83 edges)
Layer 6 — Baseline detectors (IsolationForest / OCSVM / Autoencoder)
Layer 5 — TRTS Random Forest (windowed 30-feat reference for §7)
Layer 4 — XGBoost Hybrid Guardian ★ production detector
Layer 3 — Attack injection engine (SP/CV/PV × bias/freeze/replay)
Layer 2 — 5 GRU controllers (PC, LC, FC, TC, CC)
Layer 1 — GRU plant surrogate (5.25 M params, 2×512 encoder-decoder)
```

Tab 1 (Attack Simulator) drives layers 1+2+3 live.
Tab 2 (Rollout Tester) reads precomputed eval JSON from layers 1+4_evaluate.
Tab 3 (Loop Explorer) visualises layers 1+7.
Tab 4 (Classifier Validation) reports layer 4 with the 4-experiment matrix.

---

## QA & verification

A 10-phase audit was run prior to deployment. Final result:

| Phase | Coverage | Result |
|---|---|---|
| 1 — Artifact integrity | 21 critical files exist + parse | ✅ |
| 2 — Header values | 5 metric cards trace to source | ✅ |
| 3 — Tab 1 attack physics | 9 (loop × injection × type) permutations + determinism | ✅ |
| 4 — Tab 2 NRMSE bars | 40 cells (5 PVs × 4 scenarios × 2 horizons) | ✅ |
| 5 — Tab 3 loop math | KS values + radar formula correct for all 5 loops | ✅ |
| 6 — Tab 4 classifier | A/B/C/D F1+P+R+AUROC valid | ✅ |
| 7 — Cross-tab consistency | Header F1 = Tab 4 D · F1, etc. | ✅ |
| 8 — Logical sanity | Normal < AP_with, baseline < attack, etc. | ✅ |
| 9 — Streamlit AppTest | All widgets exception-free | ✅ |
| 10 — Runtime log scan | No Tracebacks / API exceptions | ✅ |

**Final tally: 147 / 149 checks passed** (the 2 "failures" are physically explainable no-ops: SP/freeze on a stationary signal produces zero deviation by definition).

---

## Project limitations (Student Guide §5)

1. **Single process** — HAI 23.05 P1 boiler only; doesn't generalise to other plants.
2. **Forecast horizon** — validated up to 30 minutes; beyond is extrapolation.
3. **Synthetic distributional fidelity** — KS test fails for 3 of 5 loops (LC, TC, CC).
4. **Threshold calibration** — dataset-specific; would need recalibration for production deployment.
5. **Single random seed** — no statistical confidence intervals; reported as point estimates.
6. **No real-time deployment validated** — the dashboard is a viewer, not a live-PLC controller.

These limitations are deliberately surfaced in the dashboard's persistent header (always visible).

---

## Citation & dataset

- **HAI 23.05** — Affiliated Institute of ETRI, Korea, 2023. Public dataset of a research-grade thermal-power-plant boiler at 1 Hz, with 52 documented attack scenarios across AP_no / AP_with / AE_no families.
- All models in this submission were trained from scratch on the team's hardware (laptop CPU + vast.ai A100 80 GB for the long-horizon plant).

---

## Contact

For questions about this submission, contact the team via the supervisor (Dr. Khaled Chahine, Electrical Engineering Department) or open an issue at https://github.com/Modhi2o2/PLANT_MIRROR_DASHBOARD.

---

*Submission prepared 2026-04-25. Both deployed services remain live indefinitely.*
