import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Paths
data_path = os.path.join("data", "cars.csv")
model_folder = "model"
plots_folder = "plots"

os.makedirs(model_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

model_file = os.path.join(model_folder, "car_price_model.pkl")


if __name__ == "__main__":
    df = pd.read_csv(data_path)

    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_file)
    print(f"Model saved at: {model_file}")

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")

    # --- Plot: Predicted vs Actual ---
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.title("Predicted vs Actual Car Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, "predicted_vs_actual.png"))
    plt.close()

    # --- Plot: Feature Importance ---
    plt.figure(figsize=(10, 6))
    plt.bar(X.columns, model.feature_importances_)
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, "feature_importance.png"))
    plt.close()

    print("Training plots saved in /plots folder.")
