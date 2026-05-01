import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# APP TITLE
# =========================
st.set_page_config(page_title="House Price AI", layout="centered")
st.title("🏡 House Price Prediction App (No Error Version)")

# =========================
# CREATE DATA INSIDE APP
# =========================
df = pd.DataFrame({
    "area": np.random.randint(500, 3500, 800),
    "bedrooms": np.random.randint(1, 6, 800),
    "bathrooms": np.random.randint(1, 4, 800),
    "age": np.random.randint(0, 40, 800),
    "location": np.random.choice(["city", "suburb", "rural"], 800)
})

df["price"] = (
    df["area"] * 120 +
    df["bedrooms"] * 50000 +
    df["bathrooms"] * 30000 -
    df["age"] * 1000 +
    np.random.randint(10000, 50000, 800)
)

# =========================
# ENCODING
# =========================
le = LabelEncoder()
df["location"] = le.fit_transform(df["location"])

X = df[["area", "bedrooms", "bathrooms", "age", "location"]]
y = df["price"]

# =========================
# TRAIN MODEL INSIDE APP
# =========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# INPUT UI
# =========================
area = st.slider("Area (sq ft)", 500, 5000, 1500)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3])
age = st.slider("Age of House", 0, 50, 10)

location_name = st.selectbox("Location", ["city", "suburb", "rural"])
location = le.transform([location_name])[0]

# =========================
# PREDICTION
# =========================
if st.button("Predict Price 💰"):
    input_data = np.array([[area, bedrooms, bathrooms, age, location]])
    prediction = model.predict(input_data)[0]

    st.success(f"🏠 Estimated Price: ₹ {int(prediction):,}")