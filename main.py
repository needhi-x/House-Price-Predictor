import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# =========================
# CREATE FOLDERS
# =========================
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("images", exist_ok=True)

# =========================
# DATASET (SYNTHETIC)
# =========================
np.random.seed(42)

df = pd.DataFrame({
    "area": np.random.randint(500, 3500, 1000),
    "bedrooms": np.random.randint(1, 6, 1000),
    "bathrooms": np.random.randint(1, 4, 1000),
    "age": np.random.randint(0, 40, 1000),
    "location": np.random.choice(["city", "suburb", "rural"], 1000)
})

# Price generation logic
df["price"] = (
    df["area"] * 120 +
    df["bedrooms"] * 50000 +
    df["bathrooms"] * 30000 -
    df["age"] * 1000 +
    np.random.randint(10000, 50000, 1000)
)

# Save dataset
df.to_csv("data/housing_data.csv", index=False)

# =========================
# ENCODING
# =========================
le = LabelEncoder()
df["location"] = le.fit_transform(df["location"])

# Save encoder mapping (IMPORTANT for Streamlit)
location_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
joblib.dump(location_mapping, "models/location_mapping.pkl")

# =========================
# FEATURES / TARGET
# =========================
X = df[["area", "bedrooms", "bathrooms", "age", "location"]]
y = df["price"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODELS
# =========================
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# =========================
# EVALUATION
# =========================
print("\nLinear Regression")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R2:", r2_score(y_test, lr_pred))

print("\nRandom Forest")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2:", r2_score(y_test, rf_pred))

# =========================
# SAVE MODEL (IMPORTANT)
# =========================
joblib.dump(rf, "models/house_price_model.pkl")

print("\nModel saved successfully!")