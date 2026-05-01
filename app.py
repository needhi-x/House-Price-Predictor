import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD MODEL + DATA
# =========================
model = joblib.load("models/house_price_model.pkl")
location_map = joblib.load("models/location_mapping.pkl")

df = pd.read_csv("data/housing_data.csv")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="House Price AI", layout="wide")

st.title("🏡 AI House Price Prediction System")
st.markdown("### Predict property prices + explore data insights")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Enter House Details")

area = st.sidebar.slider("Area (sq ft)", 500, 5000, 1500)
bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3])
age = st.sidebar.slider("Age of House", 0, 50, 10)

location_name = st.sidebar.selectbox("Location", list(location_map.keys()))
location = location_map[location_name]

# =========================
# PREDICTION
# =========================
input_data = np.array([[area, bedrooms, bathrooms, age, location]])

prediction = model.predict(input_data)[0]

st.subheader("🏠 Prediction Result")
st.success(f"Estimated House Price: ₹ {int(prediction):,}")

# =========================
# TABS DASHBOARD
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visual Insights", "📉 Model Insights"])

# -------------------------
# TAB 1: DATA
# -------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

# -------------------------
# TAB 2: VISUALS
# -------------------------
with tab2:
    st.subheader("Price Distribution")

    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# -------------------------
# TAB 3: MODEL INSIGHTS
# -------------------------
with tab3:
    st.subheader("Feature Importance (Random Forest)")

    features = ["area", "bedrooms", "bathrooms", "age", "location"]
    importance = model.feature_importances_

    fig3, ax3 = plt.subplots()
    ax3.barh(features, importance)
    st.pyplot(fig3)

    st.write("Higher value = more impact on price")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Built with Machine Learning + Streamlit | Portfolio Project")