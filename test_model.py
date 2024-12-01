import pickle

# Attempt to load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    # Test a prediction
    print("Prediction:", model.predict(["Test article"]))
except Exception as e:
    print("Error:", e)
