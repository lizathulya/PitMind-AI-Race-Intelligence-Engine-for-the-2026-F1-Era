# ⚡ PitMind — AI Race Intelligence Engine for the 2026 F1 Era

> *Fusing time series anomaly detection, XGBoost, and LLM-powered narrative generation to decode F1 race strategy in real time.*

---

## 📌 Overview

PitMind is an end-to-end AI system built around the 2026 Formula 1 regulatory era — the most complex strategic environment in F1 history. With 50/50 hybrid power, active aerodynamics replacing DRS, and driver-controlled energy boost modes, race strategy is now a **real-time resource allocation problem**.

PitMind models this environment across three intelligent modules:

- **Energy Battle Detector** — identifies energy attack and recharge phases from lap telemetry using changepoint detection
- **Overtake Probability Model** — predicts overtake windows using 2026-aware features like energy advantage, active aero state, and a custom Driver Aggression Index
- **LLM Narrative Engine** — grounds an LLM in telemetry + transcribed team radio to explain *why* performance shifts happen, not just *that* they happen

---

## 🛠️ Tech Stack

- **Languages:** Python, SQL
- **Data:** `fastf1` (official F1 telemetry API), F1 Live Timing API (team radio)
- **Transcription:** OpenAI Whisper (local, free)
- **Time Series:** `ruptures` (changepoint detection), rolling z-scores
- **ML Model:** XGBoost + SHAP explainability
- **LLM:** Anthropic Claude API (grounded narrative generation)
- **Storage:** SQLite with window functions & multi-table JOINs
- **Deployment:** Streamlit
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
PitMind/
├── ingest.py                  # fastf1 + Whisper data pipeline → SQLite
├── detect_anomalies.py        # Energy battle & changepoint detection
├── model.py                   # XGBoost overtake classifier + SHAP
├── narrative.py               # LLM prompt builder & narrative generator
├── app.py                     # Streamlit race replay UI
├── schema.sql                 # Full SQLite schema
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_feature_eng.ipynb   # Feature engineering & aggression index
│   └── 03_modelling.ipynb     # Model training, evaluation & SHAP
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

**1️⃣ Clone the repository:**
```bash
git clone https://github.com/yourusername/PitMind.git
cd PitMind
```

**2️⃣ Install dependencies:**
```bash
pip install -r requirements.txt
```

**3️⃣ Ingest race data (populates SQLite DB + transcribes team radio):**
```bash
python ingest.py
```
> ⏱️ First run takes ~10–15 min per race session due to Whisper transcription. Audio clips are cached locally after first download.

**4️⃣ Run anomaly detection & train the model:**
```bash
python detect_anomalies.py
python model.py
```

**5️⃣ Launch the Streamlit app:**
```bash
streamlit run app.py
```

---

## 🧠 System Architecture

```
fastf1 API
    ├── Lap telemetry ──────────────────► Changepoint Detection
    │   (times, sectors, tire age)              │
    │                                     Anomaly Tagging
    └── Team radio ──► Whisper ──► DB          │
                       transcripts             │
                            │                  ▼
                            └──────► LLM Prompt Builder
                                           │
                                           ▼
                                  Narrative + Explanation
                                           │
                                           ▼
                                  Streamlit Race Replay UI
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Overtake prediction accuracy | ~XX% |
| Precision (overtake class) | ~XX% |
| Recall (overtake class) | ~XX% |
| F1 Score | ~XX% |

> Results across 8 race sessions from the 2025 season used as a pre-2026 baseline.

Top SHAP features driving predictions:
- `pace_delta` — lap time gap between battling drivers
- `estimated_energy_advantage` — inferred from pace surge patterns
- `battle_duration_laps` — pressure buildup over time
- `driver_aggression_index` — custom behavioral feature (see below)

---

## 🔥 Signature Feature: Driver Aggression Index

A custom metric computed per driver across a full season:

| Component | Weight |
|---|---|
| Attack rate (attempts per opportunity) | 40% |
| Risk score (attempts from disadvantaged position) | 40% |
| Energy aggression (how early in charge cycle attacks occur) | 20% |

Visualised as a per-driver radar chart in the Streamlit app. Turns model outputs into driver *characterisations*, not just predictions.

---

## ✨ Key Learnings

- Framed energy deployment decisions as a **Markov Decision Process** — state, action, reward across each lap
- Built a **multi-source data pipeline**: telemetry + audio → transcription → structured DB
- Applied **changepoint detection** (`ruptures`) to identify energy attack/recharge phases in pace series
- Used **SHAP values** to explain individual overtake predictions, not just global feature importance
- Engineered **2026-aware features** (active aero state, energy advantage proxy) that have no equivalent in pre-2026 F1 datasets
- Built a **grounded LLM pipeline** — model only generates narratives backed by telemetry + radio evidence, eliminating hallucination risk
- Designed a **normalised SQLite schema** with window functions to power the entire downstream pipeline from a single JOIN query
