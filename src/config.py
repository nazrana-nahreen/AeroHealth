"""
AeroHealth AI - Configuration and Constants
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset columns
INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Meaningful sensor descriptions for the dashboard & reporting
SENSOR_NAMES = {
    "sensor_1": "Fan Inlet Temp (T2) [°R]",
    "sensor_2": "LPC Outlet Temp (T24) [°R]",
    "sensor_3": "HPC Outlet Temp (T30) [°R]",
    "sensor_4": "LPT Outlet Temp (T50) [°R]",
    "sensor_5": "Fan Inlet Pressure (P2) [psia]",
    "sensor_6": "Bypass Duct Pressure (P15) [psia]",
    "sensor_7": "HPC Outlet Pressure (P30) [psia]",
    "sensor_8": "Physical Fan Speed (Nf) [rpm]",
    "sensor_9": "Physical Core Speed (Nc) [rpm]",
    "sensor_10": "Engine Pressure Ratio (epr) [P50/P2]",
    "sensor_11": "HPC Outlet Static Pressure (Ps30) [psia]",
    "sensor_12": "Ratio of Fuel Flow to Ps30 (phi) [pps/psi]",
    "sensor_13": "Corrected Fan Speed (NRf) [rpm]",
    "sensor_14": "Corrected Core Speed (NRc) [rpm]",
    "sensor_15": "Bypass Ratio (BPR)",
    "sensor_16": "Burner Fuel-Air Ratio (farB)",
    "sensor_17": "Bleed Enthalpy (htBleed)",
    "sensor_18": "Demanded Fan Speed (Nf_dmd) [rpm]",
    "sensor_19": "Demanded Corrected Fan Speed (PCNfR_dmd) [rpm]",
    "sensor_20": "HPT Coolant Bleed (W31) [lbm/s]",
    "sensor_21": "LPT Coolant Bleed (W32) [lbm/s]",
}

# Low-variance / constant sensors in FD001 that provide no predictive value
FD001_DROP_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10", 
    "sensor_16", "sensor_18", "sensor_19"
]

# Standard RUL clipping cap (cycles)
DEFAULT_RUL_CAP = 125

# Alert thresholds (Remaining Useful Life in cycles)
ALERT_NORMAL_THRESHOLD = 75   # > 75 cycles: Healthy
ALERT_CAUTION_THRESHOLD = 30  # 30-75 cycles: Maintenance scheduled
ALERT_CRITICAL_THRESHOLD = 0  # < 30 cycles: Urgent inspection required
