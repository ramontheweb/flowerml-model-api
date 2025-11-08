# test_model.py
import joblib
import json

# Load the model and metadata
model = joblib.load('iris_classifier.joblib')
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

print("🧪 Testing the trained model...")
print(f"Model Type: {metadata['model_type']}")
print(f"Accuracy: {metadata['accuracy']:.4f}")

# Test a new sample
test_sample = [[5.1, 3.5, 1.4, 0.2]]  # setosa
prediction = model.predict(test_sample)
probability = model.predict_proba(test_sample)

predicted_class = metadata['target_names'][prediction[0]]
confidence = probability[0][prediction[0]]

print(f"\nTest Sample: {test_sample[0]}")
print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
print("✅ Model is working correctly!")
