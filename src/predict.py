import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/har_model_v1.pkl")

# Load test data (or new unseen data)
data = pd.read_csv("test.csv")

# Separate features
X = data.drop(columns=["Activity"])

# Predict
predictions = model.predict(X)

print("\nFirst 10 Predictions:")
print(predictions[:10])
