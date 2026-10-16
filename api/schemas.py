"""
AeroHealth AI - Pydantic Request & Response Schemas
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class SensorCycleReading(BaseModel):
    unit_number: Optional[int] = Field(default=1, description="Engine unit identifier")
    time_in_cycles: int = Field(..., description="Operating cycle number")
    setting_1: float = Field(..., description="Altitude operational setting")
    setting_2: float = Field(..., description="Mach number operational setting")
    setting_3: float = Field(default=100.0, description="Throttle resolver angle")
    
    # 21 sensors
    sensor_1: float = Field(default=518.67)
    sensor_2: float = Field(...)
    sensor_3: float = Field(...)
    sensor_4: float = Field(...)
    sensor_5: float = Field(default=14.62)
    sensor_6: float = Field(default=21.61)
    sensor_7: float = Field(...)
    sensor_8: float = Field(...)
    sensor_9: float = Field(...)
    sensor_10: float = Field(default=1.3)
    sensor_11: float = Field(...)
    sensor_12: float = Field(...)
    sensor_13: float = Field(...)
    sensor_14: float = Field(...)
    sensor_15: float = Field(...)
    sensor_16: float = Field(default=0.03)
    sensor_17: float = Field(...)
    sensor_18: float = Field(default=2388.0)
    sensor_19: float = Field(default=100.0)
    sensor_20: float = Field(...)
    sensor_21: float = Field(...)


class SequencePredictionRequest(BaseModel):
    unit_id: Optional[int] = Field(default=1, description="Engine Unit identifier")
    readings: List[SensorCycleReading] = Field(
        ..., 
        min_length=1, 
        description="Historical sequence of cycles (at least 1, recommended >= 5 for rolling window)"
    )


class SinglePredictionResponse(BaseModel):
    unit_id: int
    latest_cycle: int
    predicted_rul: float
    health_index_pct: float
    alert_level: str
    recommendation: str
    model_version: str


class ModelMetadataResponse(BaseModel):
    model_name: str
    training_dataset: str
    features_count: int
    metrics: Dict[str, float]
    status: str
