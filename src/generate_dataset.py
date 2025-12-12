import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Folders
data_folder = "data"
plots_folder = "plots"

os.makedirs(data_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

csv_path = os.path.join(data_folder, "cars.csv")

def generate_car_dataset(n=1000):
    np.random.seed(42)

    car_size = np.random.randint(1000, 5000, n)
    mileage = np.random.randint(5000, 200000, n)
    age = np.random.randint(1, 20, n)
    brand_factor = np.random.choice([1.0, 1.2, 1.4, 1.6], n)
    
    price = (car_size * 120) + (mileage * -0.2) + (age * -1500) + (brand_factor * 8000)
    price += np.random.normal(0, 5000, n)

    df = pd.DataFrame({
        "car_size": car_size,
        "mileage": mileage,
        "age": age,
        "brand_factor": brand_factor,
        "price": price.astype(int)
    })

    return df


if __name__ == "__main__":
    df = generate_car_dataset()
    df.to_csv(csv_path, index=False)

    print(f"Dataset saved to: {csv_path}")

    # --- Plot: Mileage vs Price ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df["mileage"], df["price"], alpha=0.4)
    plt.title("Mileage vs Price")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, "mileage_vs_price.png"))
    plt.close()

    # --- Plot: Age vs Price ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df["age"], df["price"], alpha=0.4)
    plt.title("Age vs Price")
    plt.xlabel("Car Age (years)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, "age_vs_price.png"))
    plt.close()

    print("Plots saved inside /plots folder.")
