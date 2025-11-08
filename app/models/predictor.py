# app/models/predictor.py
import joblib
import numpy as np
from typing import Dict, Any
import json

class IrisPredictor:
    """
    A class to handle Iris flower predictions using the trained model.
    """
    
    def __init__(self, model_path: str = "iris_classifier.joblib"):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved joblib model file
        """
        try:
            self.model = joblib.load(model_path)
            # Load metadata for class names and feature info
            with open('model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            self.class_names = self.metadata['target_names']
            self.feature_names = self.metadata['feature_names']
            self.is_loaded = True
            print(f"✅ Model loaded successfully. Classes: {self.class_names}")
        except Exception as e:
            self.is_loaded = False
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, features: list) -> Dict[str, Any]:
        """
        Make prediction and return detailed results.
        
        Args:
            features: List of 4 feature values [sepal_length, sepal_width, petal_length, petal_width]
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded properly")
        
        # Convert to numpy array and ensure correct shape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features_array)
        probabilities = self.model.predict_proba(features_array)
        
        # Get results
        class_idx = prediction[0]
        confidence = probabilities[0][class_idx]
        
        return {
            "predicted_class": self.class_names[class_idx],
            "confidence": float(confidence),
            "class_probabilities": {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            },
            "all_predictions": [
                {
                    "class": self.class_names[i],
                    "probability": float(prob)
                }
                for i, prob in enumerate(probabilities[0])
            ]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model"""
        return {
            "model_type": self.metadata.get('model_type', 'Unknown'),
            "accuracy": self.metadata.get('accuracy', 0),
            "feature_names": self.feature_names,
            "target_names": self.class_names,
            "training_samples": self.metadata.get('training_samples', 0),
            "test_samples": self.metadata.get('test_samples', 0),
            "timestamp": self.metadata.get('timestamp', 'Unknown')
        }
