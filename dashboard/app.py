"""
AeroHealth AI - Predictive Maintenance Web Dashboard
Streamlit-based aircraft fleet monitoring & RUL estimation interface.
"""
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODELS_DIR,
    SENSOR_NAMES,
    ALERT_NORMAL_THRESHOLD,
    ALERT_CAUTION_THRESHOLD,
)
from src.data_loader import load_raw_dataset, load_ground_truth_rul
from src.feature_engineering import extract_rolling_features

# Streamlit Page Setup
st.set_page_config(
    page_title="AeroHealth AI | Aircraft Engine Health Cockpit",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #30363D;
    }
    .badge-normal {
        background-color: #238636;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-caution {
        background-color: #D29922;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-critical {
        background-color: #DA3633;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    model_path = MODELS_DIR / "aerohealth_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def get_fleet_data():
    test_df = load_raw_dataset("FD001", "test")
    true_rul = load_ground_truth_rul("FD001")
    return test_df, true_rul


pipeline = load_pipeline()
test_df, true_rul_series = get_fleet_data()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("✈️ AeroHealth AI")
    st.caption("Aircraft Engine Predictive Maintenance System")
    st.divider()

    st.subheader("Fleet Unit Selector")
    available_units = sorted(test_df["unit_number"].unique().tolist())
    selected_unit = st.selectbox(
        "Select Engine Unit ID",
        options=available_units,
        index=0,
        help="Select an aircraft turbofan engine from the fleet to monitor.",
    )

    unit_history = test_df[test_df["unit_number"] == selected_unit].sort_values("time_in_cycles")
    max_recorded_cycle = int(unit_history["time_in_cycles"].max())

    st.subheader("Simulation Control")
    sim_cycle = st.slider(
        "Observation Cycle",
        min_value=5,
        max_value=max_recorded_cycle,
        value=max_recorded_cycle,
        help="Simulate historical points in this engine's operating lifecycle.",
    )

    st.divider()
    if pipeline:
        st.caption(f"**Active Model:** {pipeline['model_name']}")
        st.caption(f"**Test RMSE:** {pipeline['metrics']['rmse']} cycles")
        st.caption(f"**NASA Score:** {pipeline['metrics']['nasa_score']}")
    else:
        st.warning("⚠️ Model artifact not loaded. Train model first.")

# ----------------- MAIN COCKPIT -----------------
st.title(f"Turbofan Engine #{selected_unit} Health Dashboard")
st.markdown("Real-time prognostics & health management (PHM) telemetric view.")

# Slice data up to the selected observation cycle
observed_df = unit_history[unit_history["time_in_cycles"] <= sim_cycle].copy()

# Perform Inference
if pipeline and len(observed_df) >= 5:
    feat_df = extract_rolling_features(observed_df, window=pipeline.get("window", 5))
    last_row = feat_df.iloc[[-1]]
    X_scaled = pd.DataFrame(
        pipeline["scaler"].transform(last_row[pipeline["feature_cols"]]),
        columns=pipeline["feature_cols"],
    )
    pred_rul = float(np.clip(pipeline["model"].predict(X_scaled)[0], 0, pipeline["rul_cap"]))
else:
    pred_rul = 0.0

# True RUL calculation for selected cycle
actual_final_rul = true_rul_series.get(selected_unit, 0)
actual_current_rul = (max_recorded_cycle - sim_cycle) + actual_final_rul
health_pct = min(100.0, max(0.0, (pred_rul / 125.0) * 100.0))

# Status Alert
if pred_rul >= ALERT_NORMAL_THRESHOLD:
    status_text = "NORMAL"
    status_badge = '<span class="badge-normal">NORMAL OPERATION</span>'
    rec_text = "All engine sensor parameters are nominal. Standard scheduled maintenance applies."
elif pred_rul >= ALERT_CAUTION_THRESHOLD:
    status_text = "CAUTION"
    status_badge = '<span class="badge-caution">MAINTENANCE CAUTION</span>'
    rec_text = "Core degradation signature detected. Schedule workshop inspection within 20 cycles."
else:
    status_text = "CRITICAL"
    status_badge = '<span class="badge-critical">CRITICAL ALERT</span>'
    rec_text = "Accelerated component wear detected! Immediate engine overhaul required before next sortie."

# Top KPI Metrics Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Observed Cycle", f"{sim_cycle} / {max_recorded_cycle}")
with col2:
    st.metric(
        "Estimated RUL",
        f"{pred_rul:.1f} cycles",
        delta=f"Actual: {actual_current_rul} cycles",
        delta_color="off",
    )
with col3:
    st.metric("Health Index", f"{health_pct:.1f}%")
with col4:
    st.markdown("**Maintenance Status**")
    st.markdown(status_badge, unsafe_allow_html=True)

st.info(f"📋 **Action Recommendation:** {rec_text}")

# Visual Progress of Remaining Useful Life
st.markdown("#### Engine Health State")
st.progress(health_pct / 100.0)

# ----------------- SENSOR TREND CHARTS -----------------
st.subheader("Telemetry & Sensor Degradation Trajectories")

# Sensor selector for plotting
key_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_15"]
selected_sensors = st.multiselect(
    "Select Sensors to Graph",
    options=key_sensors,
    default=["sensor_3", "sensor_4", "sensor_11"],
    format_func=lambda s: f"{s} ({SENSOR_NAMES.get(s, s)})",
)

if selected_sensors:
    chart_data = observed_df.set_index("time_in_cycles")[selected_sensors]
    # Rename columns with friendly sensor names
    chart_data = chart_data.rename(columns={s: SENSOR_NAMES.get(s, s) for s in selected_sensors})
    st.line_chart(chart_data)

# ----------------- RAW TELEMETRY INSPECTION -----------------
with st.expander("🔍 Inspect Raw Recent Flight Telemetry Data"):
    st.dataframe(observed_df.tail(10), use_container_width=True)
