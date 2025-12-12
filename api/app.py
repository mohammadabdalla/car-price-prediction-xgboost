from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load the trained XGBoost model
model_path = os.path.join("model", "car_price_model.pkl")
model = joblib.load(model_path)

app = FastAPI(title="Car Price Prediction API")

# Pydantic model for request body
class CarFeatures(BaseModel):
    car_size: int
    mileage: int
    age: int
    brand_factor: float

# Simple health check endpoint
@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running 🚗"}

# Prediction endpoint
@app.post("/predict")
def predict_price(features: CarFeatures):
    # Convert input to numpy array and reshape for model
    data = np.array([
        features.car_size,
        features.mileage,
        features.age,
        features.brand_factor
    ]).reshape(1, -1)

    # Predict
    prediction = model.predict(data)[0]

    return {
        "predicted_price": float(prediction)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # just 'app' because we're inside api/
        host="0.0.0.0",
        port=8000,
        reload=True
    )
