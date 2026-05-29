# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BusFreq Madrid** — a Master's thesis (TFM) building a dynamic bus frequency optimization system for EMT Madrid. The system predicts public transport demand 24–48 hours ahead and recommends optimal bus frequencies, targeting 10–15 representative lines.

Full specification: `tfm_busfreq_madrid.md`

## Architecture

Three independent but sequentially dependent modules:

```
EMT API / AEMET / GTFS / Events
        ↓
1. Demand Prediction   →  2. Frequency Optimizer  →  3. Dashboard (React + FastAPI)
   (LightGBM/Prophet/LSTM)    (OR-Tools/PuLP ILP)       (Recharts + Leaflet)
```

### Module 1 — Demand Prediction (`/demand/`)
- Forecasts validations per line/stop/time slot, 24–48h horizon
- Walk-forward temporal cross-validation (no future leakage)
- Features: calendar flags, holidays, AEMET weather, event indicators, temporal lags
- Models: LightGBM (baseline), Prophet (seasonal), LSTM (PyTorch)
- Experiment tracking via MLflow; feature importance via SHAP

### Module 2 — Frequency Optimizer (`/optimizer/`)
- Integer Linear Programming: minimizes wait time subject to fleet budget, min frequency, and driver shift constraints
- Input: demand forecasts from Module 1
- Output: recommended frequencies for next 24–48h + cost delta vs. current schedule
- Library: OR-Tools (preferred) or PuLP

### Module 3 — Dashboard (`/dashboard/`)
- **Backend:** FastAPI serving predictions and optimization results
- **Frontend:** React with Recharts (charts) and Leaflet (Madrid map heatmap)
- Key features: scenario simulation (weather/events/special days), current vs. recommended frequency comparison, wait time impact, cost analysis
- Deployment: Docker + Azure App Service

## Data Sources

All public and free:
- **EMT Madrid Open API** — validations by line/stop/time slot (requires API key registration)
- **GTFS** — network topology and static schedules
- **AEMET OpenData API** — historical and forecast weather (requires API key)
- **Madrid Open Data** — public events, local holidays

Raw data goes in `/data/raw/`, processed data in `/data/processed/`.

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Data | Python, pandas, requests |
| ML | LightGBM, Prophet, PyTorch |
| Explainability | SHAP |
| Tracking | MLflow |
| Optimization | OR-Tools / PuLP |
| Backend | FastAPI, uvicorn |
| Frontend | React, Recharts, Leaflet |
| Infra | Docker, Azure App Service |

## Development Setup (planned)

```bash
# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Backend dev server
uvicorn dashboard.api.main:app --reload

# Frontend dev server
cd dashboard/frontend && npm install && npm run dev

# Run MLflow UI
mlflow ui

# Train demand models
python demand/train.py --config configs/lgbm.yaml

# Run optimizer
python optimizer/optimize.py --date 2025-06-01
```

## Key Design Decisions

- **Walk-forward validation only** — never use future data in training folds; standard k-fold is inappropriate for time series
- **Pilot scope** — 10–15 lines selected by demand variability and geographic diversity, not all EMT lines
- **ILP for optimization** — exact solver preferred over heuristics given the small pilot fleet size
- **EMT API rate limits** — cache raw API responses locally before any processing; data collection is a bottleneck
