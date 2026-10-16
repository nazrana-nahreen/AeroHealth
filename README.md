# ✈️ AeroHealth AI: Aircraft Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AeroHealth AI** is an end-to-end Machine Learning and Prognostics & Health Management (PHM) system for aircraft turbofan engines. It processes multivariate time-series sensor telemetry, estimates degradation trajectories and Remaining Useful Life (RUL), and delivers actionable maintenance intelligence via a high-performance REST API and an interactive cockpit dashboard.

---

## 🌟 Key Features

- **Ingestion & Data Pipeline**: Parses NASA C-MAPSS run-to-failure trajectories (`FD001` through `FD004`) with schema validation and piecewise linear RUL target labeling (125-cycle cap).
- **Temporal Feature Engineering**: Filters low-variance sensors and extracts rolling statistics (mean, standard deviation) over operating cycles to capture degradation momentum.
- **Model Benchmarking**:
  - Baseline Ridge Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **NASA Asymmetric Scoring**: Evaluates models using the official NASA PHM 2008 scoring function, which penalizes late maintenance predictions exponentially higher than early predictions.
- **Production REST API (FastAPI)**:
  - Real-time RUL estimation and health index calculation (0–100%).
  - Multi-tier alert classification (`NORMAL`, `CAUTION`, `CRITICAL`).
  - Fleet inspection endpoints.
- **Interactive Cockpit Dashboard (Streamlit)**:
  - Fleet engine selector (Units 1–100).
  - Historical time-travel slider to simulate in-flight progressive degradation.
  - Multi-sensor telemetry degradation curves.
- **Testing & Verification**: Automated unit and integration test suite (`pytest`).

---

## 🏗️ Architecture & Project Structure

```text
AeroHealth/
├── data/
│   ├── raw/                 # NASA C-MAPSS dataset files (FD001 - FD004)
│   └── processed/           # Processed datasets
├── models/
│   └── aerohealth_model.joblib # Serialized model pipeline & scaler
├── src/
│   ├── config.py            # Global paths, sensor metadata, thresholds
│   ├── data_loader.py       # Ingestion & RUL label generation
│   ├── feature_engineering.py # Rolling window stats & scaling
│   ├── evaluate.py          # RMSE, MAE, R2 & NASA PHM scoring function
│   └── train.py             # Model training & benchmarking pipeline
├── api/
│   ├── app.py               # FastAPI REST API service
│   └── schemas.py           # Pydantic request/response schemas
├── dashboard/
│   └── app.py               # Streamlit fleet cockpit dashboard
├── tests/
│   ├── test_data_loader.py  # Data pipeline unit tests
│   └── test_api.py          # FastAPI integration tests
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train and Benchmark Models
Run the automated training and evaluation pipeline:
```bash
python src/train.py
```
This trains candidate models on `FD001`, computes RMSE and NASA PHM scores against the test ground truth, and serializes the top-performing pipeline into `models/aerohealth_model.joblib`.

### 3. Launch REST API
Start the FastAPI inference service:
```bash
uvicorn api.app:app --reload --port 8000
```
- Interactive Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 4. Launch Fleet Dashboard
Start the Streamlit cockpit:
```bash
streamlit run dashboard/app.py
```
Access the dashboard at [http://localhost:8501](http://localhost:8501).

---

## 🧪 Running Tests
Execute the unit and integration tests:
```bash
pytest tests/ -v
```

---

## 📊 Evaluation: NASA Asymmetric Scoring Function

In aviation predictive maintenance, an optimistic prediction (predicting an engine will last longer than it actually does) leads to catastrophic failure during flight. Conversely, an early prediction only results in slightly premature scheduled maintenance.

The NASA PHM 2008 asymmetric penalty function reflects this operational reality:

$$d = \hat{y} - y \quad (\text{Predicted RUL} - \text{True RUL})$$

$$S = \sum_{i=1}^{N} s_i \quad \text{where} \quad s_i = \begin{cases} e^{-d/13} - 1 & \text{if } d < 0 \text{ (Early prediction)} \\ e^{d/10} - 1 & \text{if } d \ge 0 \text{ (Late prediction)} \end{cases}$$

---

## 🛡️ Responsible Use & Aviation Notice
This project is an educational and engineering prototype. Predictions must not be used for actual aircraft maintenance sign-offs or flight safety dispatch decisions.
