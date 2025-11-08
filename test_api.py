# test_api.py
import requests
import json

BASE_URL = "http://localhost:8000"

def test_all_endpoints():
    print("🧪 Testing Iris Classification API Endpoints...")
    print("=" * 50)
    
    try:
        # Test root endpoint
        print("1. Testing root endpoint...")
        response = requests.get(f"{BASE_URL}/")
        print(f"   ✅ Status: {response.status_code}")
        print(f"   📝 Response: {response.json()}")
        print()
        
        # Test health check
        print("2. Testing health check...")
        response = requests.get(f"{BASE_URL}/health")
        print(f"   ✅ Status: {response.status_code}")
        health_data = response.json()
        print(f"   🩺 Status: {health_data['status']}")
        print(f"   🤖 Model Loaded: {health_data['model_loaded']}")
        print()
        
        # Test model info
        print("3. Testing model info...")
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"   ✅ Status: {response.status_code}")
        model_info = response.json()
        print(f"   🧠 Model Type: {model_info['model_type']}")
        print(f"   📊 Accuracy: {model_info['accuracy']:.4f}")
        print(f"   🌸 Classes: {model_info['target_names']}")
        print()
        
        # Test examples
        print("4. Testing examples endpoint...")
        response = requests.get(f"{BASE_URL}/examples")
        print(f"   ✅ Status: {response.status_code}")
        examples = response.json()
        print(f"   📋 Available examples: {list(examples.keys())}")
        print()
        
        # Test predictions with different examples
        test_cases = [
            ("setosa", examples['setosa_example']),
            ("versicolor", examples['versicolor_example']),
            ("virginica", examples['virginica_example'])
        ]
        
        print("5. Testing predictions...")
        for flower_type, test_data in test_cases:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_data
            )
            print(f"   🌸 {flower_type.capitalize()} example:")
            print(f"      ✅ Status: {response.status_code}")
            result = response.json()
            print(f"      🔮 Prediction: {result['predicted_class']}")
            print(f"      💯 Confidence: {result['confidence']:.4f}")
            print()
        
        print("🎉 All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running!")
        print("💡 Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_all_endpoints()
