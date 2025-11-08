# app/schemas.py
from pydantic import BaseModel, Field
from typing import Dict

class IrisFeatures(BaseModel):
    """
    Input features for Iris flower classification.
    All measurements are in centimeters.
    """
    sepal_length: float = Field(..., ge=4.0, le=8.0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=2.0, le=4.5, description="Sepal width in cm")
    petal_length: float = Field(..., ge=1.0, le=7.0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0.1, le=2.5, description="Petal width in cm")
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    timestamp: str = None

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str
    accuracy: float
    feature_names: list
    target_names: list
    training_samples: int
    test_samples: int
