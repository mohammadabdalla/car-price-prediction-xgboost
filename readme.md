# 🚗 Car Price Prediction API (XGBoost)

A complete end-to-end machine learning project for predicting car prices using **XGBoost** and exposing the trained model through a **FastAPI REST API**.

This project includes:

- Synthetic dataset generation
- Data preprocessing
- Model training (XGBoostRegressor)
- Model saving using pickle
- Production-ready FastAPI server for predictions
- Clean & scalable folder structure

---

## 📁 Project Structure

car-price-prediction-xgboost/
├── data/
│   └── cars.csv                 # auto generated
├── model/
│   └── car_price_model.pkl      # auto generated
├── api/
│   └── app.py
├── src/
│   ├── generate_dataset.py
│   └── train_model.py
├── requirements.txt
└── README.md




---

## ⚙️ Installation

### 1️⃣ Create & activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt

🛠 Step 1 — Generate Dataset
python src/generate_dataset.py

This creates:

data/cars.csv

🧠 Step 2 — Train the Model
python src/train_model.py


This creates:

model/car_price_model.pkl

🚀 Step 3 — Start the API Server
uvicorn api.app:app --reload


The API will run at:

http://127.0.0.1:8000

📡 API Usage
➤ POST /predict

Request Body Example

{
  "year": 2018,
  "mileage": 45000,
  "brand": "Toyota",
  "engine_size": 2.0
}


Response Example

{
  "predicted_price": 17654.23
}

🧪 Testing the API with Curl
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
  "car_size": 2500,
  "mileage": 45000,
  "age": 5,
  "brand_factor": 1.4
}'

🧑‍💻 Author

Mohammad Abdullah


---