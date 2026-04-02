# TE Draft Intelligence — ML Backend

A full machine learning pipeline for predicting NFL tight end draft success, powering the TE Draft Intelligence website.

---

## Project Structure

```
te_draft_ml/
│
├── data_pipeline.py          # Data loading, cleaning, feature engineering
├── ml_models.py              # Model training, SHAP, clustering → exports JSON + pickles
├── api_server.py             # Flask REST API (reads pickled models + JSON results)
├── run.py                    # One-command launcher (train → serve)
├── requirements.txt
│
├── Copy_of_TAMIDS_Combine_Data.xlsx        ← raw data
├── Copy_of_TE_CFB_Stats_2004-2025.xlsx     ← raw data
├── Copy_of_TE_NFL_STATS_2000_2024.xlsx     ← raw data
│
├── models/                   # Pickled models (created after training)
│   ├── ensemble.pkl          # VotingClassifier (GBM + RF + Logistic)
│   ├── pick_model.pkl        # GBM draft pick regressor
│   ├── epa_model.pkl         # GBM career EPA regressor
│   ├── medians.pkl           # Feature medians for imputation
│   └── km.pkl                # KMeans archetype clusterer
│
└── results/                  # Pre-computed JSON (served by API)
    ├── players.json           # 456 historical players + predictions
    ├── prospects.json         # 27 2026 draft prospects + predictions
    └── model_info.json        # Feature importances, metrics, archetypes
```

---

## Setup

```bash
cd te_draft_ml
pip install -r requirements.txt
```

---

## Run

### Train + serve (recommended first time)
```bash
python run.py
```

### Force retrain
```bash
python run.py --train
```

### API only (models already trained)
```bash
python run.py --api
```

The API starts at **http://localhost:5050**

---

## ML Models

### 1. Success Probability (Classification)
**Goal:** Predict whether a TE will have a meaningful NFL career  
**Label:** career_rec_yds > 1,500 OR career_rec_td ≥ 10  
**Model:** Soft-voting ensemble — GBM (weight 3) + Random Forest (weight 2) + Logistic (weight 1)  
**Performance:** CV AUC ≈ 0.776 ± 0.030, Accuracy ≈ 73.6%

### 2. Draft Pick Predictor (Regression)
**Goal:** Predict which pick a prospect will be selected at  
**Model:** Gradient Boosting Regressor  
**Performance:** CV MAE ≈ 57 picks

### 3. Career EPA Regressor
**Goal:** Project total career Expected Points Added (receiving)  
**Model:** Gradient Boosting Regressor  
**Performance:** CV MAE ≈ 56 EPA

### 4. SHAP Explainability
Per-player feature attributions computed via TreeSHAP.  
Every player/prospect gets their top 6 factors explaining the model's prediction.

### 5. Prospect Archetype Clustering (KMeans k=5)
Archetypes:
- **Receiving Weapon** — High college production, elite route runner
- **Athletic Freak** — Fastest 40, best athleticism scores
- **Inline Blocker** — Heaviest, most physical
- **Move TE / H-Back** — Versatile, used in motion/split
- **Balanced Starter** — Well-rounded profile

---

## Features Used

| Feature | Description | Category |
|---|---|---|
| `40yd` | 40-yard dash time | Combine |
| `Wt` | Weight (lbs) | Combine |
| `height_in` | Height (inches) | Combine |
| `Vertical` | Vertical jump (in) | Combine |
| `Broad Jump` | Broad jump (in) | Combine |
| `3Cone` | 3-cone drill time | Combine |
| `Shuttle` | Shuttle run time | Combine |
| `pick` | Draft pick number | Draft |
| `cfb_rec_yds` | Career college receiving yards | College |
| `cfb_rec_td` | Career college receiving TDs | College |
| `cfb_rec` | Career college receptions | College |
| `cfb_ypr` | College yards per reception | College |
| `cfb_ppa_total` | College overall PPA | College |
| `cfb_ppa_pass` | College pass PPA | College |
| `cfb_usage_pass` | College pass usage rate | College |
| `cfb_rec_yds_per_season` | College rec yards per season | College |
| `cfb_rec_td_per_season` | College rec TDs per season | College |
| `cfb_rush_yds` | Career college rush yards | College |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | Health check + model metrics |
| GET | `/api/players` | All historical players (filterable) |
| GET | `/api/players?name=Kelce` | Search by name |
| GET | `/api/players?year=2011` | Filter by draft year |
| GET | `/api/prospects` | 2026 draft prospects |
| GET | `/api/model_info` | Feature importance, metrics |
| GET | `/api/player/<name>` | Single player with SHAP factors |
| GET | `/api/similar/<name>` | 8 most similar historical comps |
| GET | `/api/archetypes` | Cluster summary stats |
| POST | `/api/predict` | Custom player prediction |

### POST /api/predict — example
```json
{
  "forty": 4.52,
  "weight": 247,
  "height": 77,
  "vertical": 36.5,
  "broad": 122,
  "cone": 6.98,
  "shuttle": 4.22,
  "cfb_rec_yds": 2100,
  "cfb_rec_td": 18,
  "cfb_rec": 145,
  "cfb_ypr": 14.5,
  "cfb_ppa": 89.4,
  "cfb_usage_pass": 0.18,
  "cfb_seasons": 3
}
```

Response:
```json
{
  "ml_prob": 0.71,
  "ml_projected_pick": 34,
  "ml_projected_epa": 142.3,
  "archetype": "Receiving Weapon",
  "tier": "Elite"
}
```

---

## Connecting to the Website

The HTML frontend should point its fetch calls to `http://localhost:5050/api/...`.  

In the website JavaScript:
```js
const API = "http://localhost:5050";

// Load 2026 prospects
const prospects = await fetch(`${API}/api/prospects`).then(r => r.json());

// Custom prediction
const result = await fetch(`${API}/api/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ forty: 4.52, weight: 247, ... })
}).then(r => r.json());
```
