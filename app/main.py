# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

# Import our custom modules
from .models.predictor import IrisPredictor
from .schemas import IrisFeatures, PredictionResponse, HealthResponse, ModelInfoResponse

print("🚀 Starting Iris Classification API...")

# Initialize the predictor
try:
    predictor = IrisPredictor("iris_classifier.joblib")
    model_info = predictor.get_model_info()
    print(f"✅ Model loaded: {model_info['model_type']}")
    print(f"📊 Model accuracy: {model_info['accuracy']:.4f}")
except Exception as e:
    print(f"❌ Failed to initialize predictor: {e}")
    predictor = None

# Create FastAPI app
app = FastAPI(
    title="Iris Flower Classification API",
    description="A REST API for classifying Iris flowers using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow frontend applications to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Iris Flower Classification API",
        "status": "running",
        "docs": "/docs",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health and model status
    """
    return HealthResponse(
        status="healthy" if predictor and predictor.is_loaded else "unhealthy",
        model_loaded=predictor.is_loaded if predictor else False,
        timestamp=datetime.now().isoformat()
    )

# Model information endpoint
@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the trained model
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = predictor.get_model_info()
    return ModelInfoResponse(**info)

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    Predict Iris flower species based on input features
    
    - **sepal_length**: Sepal length in cm (4.0 - 8.0)
    - **sepal_width**: Sepal width in cm (2.0 - 4.5)  
    - **petal_length**: Petal length in cm (1.0 - 7.0)
    - **petal_width**: Petal width in cm (0.1 - 2.5)
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to list
        input_features = [
            features.sepal_length,
            features.sepal_width, 
            features.petal_length,
            features.petal_width
        ]
        
        # Make prediction
        result = predictor.predict(input_features)
        
        print(f"🔮 Prediction made: {input_features} -> {result['predicted_class']}")
        
        return PredictionResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            class_probabilities=result["class_probabilities"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Example data endpoint
@app.get("/examples")
async def get_examples():
    """
    Get example inputs for testing the API
    """
    examples = {
        "setosa_example": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "versicolor_example": {
            "sepal_length": 6.0,
            "sepal_width": 2.7,
            "petal_length": 4.2,
            "petal_width": 1.3
        },
        "virginica_example": {
            "sepal_length": 6.7,
            "sepal_width": 3.0,
            "petal_length": 5.2,
            "petal_width": 2.3
        }
    }
    return examples

print("✅ FastAPI application created successfully!")
print("📚 API documentation will be available at: http://localhost:8000/docs")

# This allows running with: python app/main.py
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
