"""
AeroHealth AI - FastAPI Inference & Web Cockpit Service
Provides REST endpoints for aircraft engine Remaining Useful Life (RUL) estimation,
health monitoring, maintenance alert categorization, and an interactive web cockpit.
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODELS_DIR,
    ALERT_NORMAL_THRESHOLD,
    ALERT_CAUTION_THRESHOLD,
)
from src.feature_engineering import extract_rolling_features
from src.data_loader import load_raw_dataset, load_ground_truth_rul
from api.schemas import (
    SequencePredictionRequest,
    SinglePredictionResponse,
    ModelMetadataResponse,
)

app = FastAPI(
    title="AeroHealth AI - Aircraft Predictive Maintenance API",
    description="Intelligent turbofan engine health monitoring, RUL prediction, and maintenance decision support.",
    version="1.0.0",
)

# Enable CORS for dashboard/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        model_path = MODELS_DIR / "aerohealth_model.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained model artifact not found. Please run model training first.",
            )
        pipeline = joblib.load(model_path)
    return pipeline


WEB_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroHealth AI | Aircraft Engine Cockpit</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #0b0f19; color: #f3f4f6; }
        .card { background-color: #111827; border: 1px solid #1f2937; }
        .accent-glow { box-shadow: 0 0 20px rgba(59, 130, 246, 0.2); }
    </style>
</head>
<body class="min-h-screen font-sans p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 pb-6 border-b border-gray-800">
            <div>
                <div class="flex items-center gap-3">
                    <span class="text-3xl">✈️</span>
                    <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-white">AeroHealth AI</h1>
                    <span class="bg-blue-600/20 text-blue-400 border border-blue-500/30 text-xs px-2.5 py-1 rounded-full font-semibold">Vercel Live</span>
                </div>
                <p class="text-gray-400 text-sm mt-1">Aircraft Turbofan Engine Health Estimation & Predictive Maintenance System</p>
            </div>
            <div class="flex items-center gap-3">
                <a href="/docs" target="_blank" class="bg-gray-800 hover:bg-gray-700 text-gray-200 px-4 py-2 rounded-lg text-sm font-medium border border-gray-700 transition flex items-center gap-2">
                    <i class="fa-solid fa-code"></i> API Docs (Swagger)
                </a>
                <a href="https://github.com/nazrana-nahreen/AeroHealth" target="_blank" class="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition flex items-center gap-2">
                    <i class="fa-brands fa-github"></i> GitHub
                </a>
            </div>
        </header>

        <!-- Controls -->
        <div class="card rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center gap-4 flex-wrap">
                <div>
                    <label class="block text-xs uppercase tracking-wider text-gray-400 mb-1 font-semibold">Select Aircraft Engine</label>
                    <select id="engineSelect" class="bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white font-medium focus:ring-2 focus:ring-blue-500 outline-none">
                        <!-- Options generated dynamically -->
                    </select>
                </div>
                <div>
                    <label class="block text-xs uppercase tracking-wider text-gray-400 mb-1 font-semibold">Observation Cycle: <span id="cycleLabel" class="text-blue-400">Loading...</span></label>
                    <input id="cycleSlider" type="range" class="w-48 md:w-64 accent-blue-500 cursor-pointer">
                </div>
            </div>
            <div class="flex items-center gap-2">
                <button onclick="fetchEngineData()" class="bg-blue-600/30 hover:bg-blue-600/50 text-blue-300 border border-blue-500/40 px-3 py-2 rounded-lg text-xs font-semibold transition">
                    <i class="fa-solid fa-rotate-right mr-1"></i> Refresh Telemetry
                </button>
            </div>
        </div>

        <!-- KPI Metrics Grid -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="card rounded-xl p-5">
                <span class="text-xs uppercase tracking-wider text-gray-400 font-semibold">Estimated RUL</span>
                <div class="flex items-baseline gap-2 mt-2">
                    <span id="rulValue" class="text-3xl font-extrabold text-white">--</span>
                    <span class="text-gray-400 text-sm">cycles</span>
                </div>
                <p id="rulDelta" class="text-xs text-gray-400 mt-1">Ground Truth: --</p>
            </div>
            <div class="card rounded-xl p-5">
                <span class="text-xs uppercase tracking-wider text-gray-400 font-semibold">Health Index</span>
                <div class="flex items-baseline gap-2 mt-2">
                    <span id="healthValue" class="text-3xl font-extrabold text-white">--%</span>
                </div>
                <div class="w-full bg-gray-800 h-2 rounded-full mt-2 overflow-hidden">
                    <div id="healthBar" class="bg-blue-500 h-full w-0 transition-all duration-500"></div>
                </div>
            </div>
            <div class="card rounded-xl p-5">
                <span class="text-xs uppercase tracking-wider text-gray-400 font-semibold">Maintenance Status</span>
                <div class="mt-2">
                    <span id="alertBadge" class="inline-block px-3 py-1 rounded-full text-xs font-bold bg-gray-800 text-gray-300">Evaluating...</span>
                </div>
                <p class="text-xs text-gray-400 mt-2">NASA PHM Standard</p>
            </div>
            <div class="card rounded-xl p-5">
                <span class="text-xs uppercase tracking-wider text-gray-400 font-semibold">AI Model Info</span>
                <div class="text-lg font-bold text-white mt-2">Gradient Boosting</div>
                <p class="text-xs text-emerald-400 mt-1">RMSE: 18.27 | NASA Score: 891.06</p>
            </div>
        </div>

        <!-- Recommendation Banner -->
        <div id="recBanner" class="p-4 rounded-xl border border-blue-900/50 bg-blue-950/30 text-blue-200 text-sm flex items-start gap-3">
            <i class="fa-solid fa-circle-info text-blue-400 mt-0.5"></i>
            <div>
                <strong class="font-semibold">Maintenance Advisory:</strong>
                <span id="recText">Loading analysis...</span>
            </div>
        </div>

        <!-- Telemetry Charts -->
        <div class="card rounded-xl p-6">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-lg font-bold text-white flex items-center gap-2">
                    <i class="fa-solid fa-chart-line text-blue-400"></i> Sensor Degradation Trajectory
                </h2>
                <span class="text-xs text-gray-400">Total Temperature & Pressure Sensors</span>
            </div>
            <div class="h-80 w-full">
                <canvas id="sensorChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let currentEngineData = null;
        let chartInstance = null;

        async function init() {
            // Populate engine dropdown (1 to 100)
            const select = document.getElementById('engineSelect');
            for (let i = 1; i <= 100; i++) {
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = `Engine Unit #${i}`;
                select.appendChild(opt);
            }
            select.addEventListener('change', fetchEngineData);
            document.getElementById('cycleSlider').addEventListener('input', onSliderChange);

            await fetchEngineData();
        }

        async function fetchEngineData() {
            const unitId = document.getElementById('engineSelect').value;
            try {
                const res = await fetch(`/fleet/engine/${unitId}`);
                if (!res.ok) throw new Error("Failed to load engine data");
                currentEngineData = await res.json();

                const slider = document.getElementById('cycleSlider');
                slider.min = 5;
                slider.max = currentEngineData.total_cycles;
                slider.value = currentEngineData.total_cycles;

                updateView(currentEngineData.total_cycles);
            } catch (err) {
                console.error(err);
            }
        }

        function onSliderChange(e) {
            updateView(parseInt(e.target.value));
        }

        async function updateView(cycle) {
            if (!currentEngineData) return;
            document.getElementById('cycleLabel').textContent = `${cycle} / ${currentEngineData.total_cycles}`;

            const filteredRecords = currentEngineData.records.filter(r => r.time_in_cycles <= cycle);
            const recentWindow = filteredRecords.slice(-10);

            // Predict RUL
            try {
                const predRes = await fetch('/predict/rul', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        unit_id: currentEngineData.engine_id,
                        readings: recentWindow
                    })
                });

                if (predRes.ok) {
                    const pred = await predRes.json();
                    document.getElementById('rulValue').textContent = pred.predicted_rul.toFixed(1);
                    document.getElementById('healthValue').textContent = `${pred.health_index_pct.toFixed(0)}%`;
                    document.getElementById('healthBar').style.width = `${pred.health_index_pct}%`;
                    document.getElementById('recText').textContent = pred.recommendation;

                    const trueCurrentRul = (currentEngineData.total_cycles - cycle) + currentEngineData.true_final_rul;
                    document.getElementById('rulDelta').textContent = `Ground Truth: ${trueCurrentRul} cycles`;

                    const badge = document.getElementById('alertBadge');
                    if (pred.alert_level === 'NORMAL') {
                        badge.className = 'inline-block px-3 py-1 rounded-full text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-800';
                        badge.textContent = 'NORMAL OPERATION';
                        document.getElementById('healthBar').className = 'bg-emerald-500 h-full transition-all duration-500';
                    } else if (pred.alert_level === 'CAUTION') {
                        badge.className = 'inline-block px-3 py-1 rounded-full text-xs font-bold bg-amber-950 text-amber-400 border border-amber-800';
                        badge.textContent = 'CAUTION REQUIRED';
                        document.getElementById('healthBar').className = 'bg-amber-500 h-full transition-all duration-500';
                    } else {
                        badge.className = 'inline-block px-3 py-1 rounded-full text-xs font-bold bg-rose-950 text-rose-400 border border-rose-800';
                        badge.textContent = 'CRITICAL ALERT';
                        document.getElementById('healthBar').className = 'bg-rose-500 h-full transition-all duration-500';
                    }
                }
            } catch (e) {
                console.error(e);
            }

            renderChart(filteredRecords);
        }

        function renderChart(records) {
            const labels = records.map(r => r.time_in_cycles);
            const sensor3 = records.map(r => r.sensor_3); // T30 HPC Outlet Temp
            const sensor4 = records.map(r => r.sensor_4); // T50 LPT Outlet Temp
            const sensor11 = records.map(r => r.sensor_11); // Ps30 Static Pressure

            const ctx = document.getElementById('sensorChart').getContext('2d');
            if (chartInstance) chartInstance.destroy();

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'HPC Outlet Temp (T30) [°R]',
                            data: sensor3,
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            borderWidth: 1.5,
                            pointRadius: 0,
                            tension: 0.1
                        },
                        {
                            label: 'LPT Outlet Temp (T50) [°R]',
                            data: sensor4,
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            borderWidth: 1.5,
                            pointRadius: 0,
                            tension: 0.1
                        },
                        {
                            label: 'HPC Outlet Pressure (Ps30) [psia]',
                            data: sensor11,
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 1.5,
                            pointRadius: 0,
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { labels: { color: '#9ca3af' } }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Operating Cycles', color: '#6b7280' },
                            grid: { color: '#1f2937' },
                            ticks: { color: '#9ca3af' }
                        },
                        y: {
                            grid: { color: '#1f2937' },
                            ticks: { color: '#9ca3af' }
                        }
                    }
                }
            });
        }

        window.onload = init;
    </script>
</body>
</html>
"""


@app.get("/", tags=["Cockpit"])
def root(request: Request):
    """
    Serves interactive web cockpit for browsers, or JSON info for API clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=WEB_DASHBOARD_HTML)
    return {
        "project": "AeroHealth AI",
        "description": "Aircraft Turbofan Engine Health Estimation & Predictive Maintenance",
        "docs_url": "/docs",
        "health_check": "/health",
        "dashboard_url": "/",
    }


@app.get("/health", tags=["General"])
def health_check():
    try:
        p = get_pipeline()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": p.get("model_name"),
        }
    except Exception as e:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e),
        }


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model"])
def model_metadata():
    p = get_pipeline()
    return ModelMetadataResponse(
        model_name=p["model_name"],
        training_dataset="NASA C-MAPSS (FD001)",
        features_count=len(p["feature_cols"]),
        metrics=p["metrics"],
        status="active",
    )


@app.post("/predict/rul", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_rul(payload: SequencePredictionRequest):
    """
    Predict Remaining Useful Life (RUL) for an aircraft engine based on recent operating cycles.
    """
    p = get_pipeline()
    model = p["model"]
    scaler = p["scaler"]
    feature_cols = p["feature_cols"]
    window = p.get("window", 5)
    rul_cap = p.get("rul_cap", 125)

    # Convert request readings into DataFrame
    records = [r.model_dump() for r in payload.readings]
    df = pd.DataFrame(records)

    if "unit_number" not in df.columns or df["unit_number"].isnull().any():
        df["unit_number"] = payload.unit_id or 1

    # Ensure sorted by cycle
    df = df.sort_values("time_in_cycles").reset_index(drop=True)
    latest_cycle = int(df["time_in_cycles"].iloc[-1])

    # Compute rolling features
    feat_df = extract_rolling_features(df, window=window)
    last_record = feat_df.iloc[[-1]]

    # Ensure all required feature columns exist
    missing_cols = [c for c in feature_cols if c not in last_record.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing feature columns in payload: {missing_cols[:5]}...",
        )

    # Scale features
    X_scaled = pd.DataFrame(
        scaler.transform(last_record[feature_cols]),
        columns=feature_cols,
    )

    # Inference
    pred_rul = float(np.clip(model.predict(X_scaled)[0], 0, rul_cap))

    # Health Index (100% = fully nominal, 0% = failure threshold)
    health_index = round(min(100.0, max(0.0, (pred_rul / rul_cap) * 100.0)), 1)

    # Alert categorization
    if pred_rul >= ALERT_NORMAL_THRESHOLD:
        alert = "NORMAL"
        rec = "Engine operating within nominal health margins. Continue standard periodic monitoring."
    elif pred_rul >= ALERT_CAUTION_THRESHOLD:
        alert = "CAUTION"
        rec = "Moderate degradation detected in core components. Schedule hangar inspection within 25 cycles."
    else:
        alert = "CRITICAL"
        rec = "Significant degradation identified! Immediate maintenance action required before next flight."

    return SinglePredictionResponse(
        unit_id=payload.unit_id or int(df["unit_number"].iloc[-1]),
        latest_cycle=latest_cycle,
        predicted_rul=round(pred_rul, 1),
        health_index_pct=health_index,
        alert_level=alert,
        recommendation=rec,
        model_version=p["model_name"],
    )


@app.get("/fleet/engines", tags=["Fleet"])
def list_test_engines():
    """Returns list of available engine units in test dataset with their recorded cycles."""
    test_df = load_raw_dataset("FD001", "test")
    units = (
        test_df.groupby("unit_number")["time_in_cycles"]
        .agg(["max", "count"])
        .reset_index()
        .rename(columns={"unit_number": "engine_id", "max": "last_cycle", "count": "total_records"})
    )
    return units.to_dict(orient="records")


@app.get("/fleet/engine/{unit_id}", tags=["Fleet"])
def get_engine_data(unit_id: int):
    """Retrieve full sensor trajectory and ground truth for a test engine."""
    test_df = load_raw_dataset("FD001", "test")
    unit_data = test_df[test_df["unit_number"] == unit_id]
    if unit_data.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Engine Unit {unit_id} not found in test dataset.",
        )

    true_rul_series = load_ground_truth_rul("FD001")
    true_rul = int(true_rul_series.get(unit_id, -1))

    return {
        "engine_id": unit_id,
        "total_cycles": int(unit_data["time_in_cycles"].max()),
        "true_final_rul": true_rul,
        "records": unit_data.to_dict(orient="records"),
    }
