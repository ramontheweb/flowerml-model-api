# train_model.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

print("🚀 Starting Iris Classification Model Training...")
print("=" * 50)

# 1. Load the Iris dataset
print("📊 Loading Iris dataset...")
iris = load_iris()

# Display dataset information
print(f"Dataset features: {iris.feature_names}")
print(f"Target names: {iris.target_names}")
print(f"Data shape: {iris.data.shape}")
print(f"Number of classes: {len(iris.target_names)}")

# 2. Prepare the data
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: 0=setosa, 1=versicolor, 2=virginica

print(f"\n📈 Sample of the data:")
print("Features (first 5 rows):")
print(X[:5])
print("Targets (first 5 rows):")
print(y[:5])

# 3. Split the data into training and testing sets
print("\n🎯 Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% of data for testing
    random_state=42,    # For reproducible results
    stratify=y          # Maintain class distribution
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Feature dimension: {X_train.shape[1]} features")

# 4. Train the model
print("\n🤖 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,   # Number of trees in the forest
    random_state=42,    # For reproducible results
    max_depth=3         # Limit tree depth to prevent overfitting
)

# Fit the model to training data
model.fit(X_train, y_train)
print("✅ Model training completed!")

# 5. Evaluate the model
print("\n📊 Evaluating model performance...")

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"📈 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Test with some example predictions
print("\n🔍 Example Predictions:")
test_samples = [
    [5.1, 3.5, 1.4, 0.2],  # Should be setosa
    [6.7, 3.0, 5.2, 2.3],  # Should be virginica
    [5.9, 2.8, 4.3, 1.2],  # Should be versicolor
]

for i, sample in enumerate(test_samples):
    prediction = model.predict([sample])
    probabilities = model.predict_proba([sample])
    predicted_class = iris.target_names[prediction[0]]
    confidence = probabilities[0][prediction[0]]
    
    print(f"Sample {i+1}: {sample}")
    print(f"  → Prediction: {predicted_class}")
    print(f"  → Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"  → All probabilities: {dict(zip(iris.target_names, [f'{p:.4f}' for p in probabilities[0]]))}")

# 7. Save the trained model
print("\n💾 Saving the trained model...")
model_filename = 'iris_classifier.joblib'
joblib.dump(model, model_filename)
print(f"✅ Model saved as: {model_filename}")

# 8. Save model metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'accuracy': float(accuracy),
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist(),
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'timestamp': pd.Timestamp.now().isoformat()
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Model metadata saved as: model_metadata.json")

# 9. Verify the saved model can be loaded
print("\n🔍 Verifying saved model...")
loaded_model = joblib.load(model_filename)
verification_accuracy = loaded_model.score(X_test, y_test)
print(f"✅ Loaded model verification accuracy: {verification_accuracy:.4f}")

print("\n" + "=" * 50)
print("🎉 Model training completed successfully!")
print(f"📁 Files created:")
print(f"   - {model_filename} (trained model)")
print(f"   - model_metadata.json (model information)")
print(f"📊 Final accuracy: {accuracy*100:.2f}%")
print("=" * 50)