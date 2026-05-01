import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="House Price Dashboard", layout="wide")

st.title("🏡 Real Estate AI Dashboard")
st.markdown("### Predict house prices + explore data insights")

# =========================
# CREATE SYNTHETIC DATA
# =========================
df = pd.DataFrame({
    "area": np.random.randint(500, 3500, 1000),
    "bedrooms": np.random.randint(1, 6, 1000),
    "bathrooms": np.random.randint(1, 4, 1000),
    "age": np.random.randint(0, 40, 1000),
    "location": np.random.choice(["city", "suburb", "rural"], 1000)
})

df["price"] = (
    df["area"] * 120 +
    df["bedrooms"] * 50000 +
    df["bathrooms"] * 30000 -
    df["age"] * 1000 +
    np.random.randint(10000, 50000, 1000)
)

# =========================
# ENCODING
# =========================
le = LabelEncoder()
df["location"] = le.fit_transform(df["location"])

X = df[["area", "bedrooms", "bathrooms", "age", "location"]]
y = df["price"]

# =========================
# MODEL TRAINING
# =========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# TABS (DASHBOARD STRUCTURE)
# =========================
tab1, tab2, tab3 = st.tabs(["🏡 Overview", "📊 Analytics", "🤖 Prediction"])

# =========================
# TAB 1 - OVERVIEW
# =========================
with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Houses", len(df))
    col2.metric("Average Price", f"₹ {int(df['price'].mean()):,}")
    col3.metric("Max Price", f"₹ {int(df['price'].max()):,}")

    st.dataframe(df.head())

# =========================
# TAB 2 - ANALYTICS
# =========================
with tab2:
    st.subheader("Price Distribution")

    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Area vs Price")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df["area"], y=df["price"], ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")

    fig3, ax3 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.subheader("Feature Importance")

    fig4, ax4 = plt.subplots()
    features = ["area", "bedrooms", "bathrooms", "age", "location"]
    ax4.barh(features, model.feature_importances_)
    st.pyplot(fig4)

# =========================
# TAB 3 - PREDICTION ENGINE
# =========================
with tab3:
    st.subheader("Enter House Details")

    area = st.slider("Area (sq ft)", 500, 5000, 1500)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3])
    age = st.slider("Age of House", 0, 50, 10)
    location_name = st.selectbox("Location", ["city", "suburb", "rural"])

    location = le.transform([location_name])[0]

    if st.button("Predict Price 💰"):
        input_data = np.array([[area, bedrooms, bathrooms, age, location]])
        prediction = model.predict(input_data)[0]

        st.success(f"🏠 Estimated House Price: ₹ {int(prediction):,}")

        st.info("This prediction is generated using a Random Forest Regression model trained on synthetic housing data.")