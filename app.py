import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# PAGE CONFIG (ZILLOW STYLE)
# =========================
st.set_page_config(
    page_title="Zillow AI | House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# =========================
# CUSTOM UI STYLE
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    h1 {
        color: #1f4e79;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🏡 Zillow AI - Real Estate Price Predictor")
st.markdown("#### AI-powered property valuation dashboard (ML + Analytics)")

# =========================
# DATA GENERATION
# =========================
df = pd.DataFrame({
    "area": np.random.randint(500, 4500, 1500),
    "bedrooms": np.random.randint(1, 6, 1500),
    "bathrooms": np.random.randint(1, 4, 1500),
    "age": np.random.randint(0, 60, 1500),
    "location": np.random.choice(["city", "suburb", "rural"], 1500)
})

df["price"] = (
    df["area"] * 140 +
    df["bedrooms"] * 50000 +
    df["bathrooms"] * 30000 -
    df["age"] * 1000 +
    np.random.randint(20000, 90000, 1500)
)

# =========================
# ENCODING
# =========================
le = LabelEncoder()
df["location"] = le.fit_transform(df["location"])

X = df[["area", "bedrooms", "bathrooms", "age", "location"]]
y = df["price"]

# =========================
# MODEL
# =========================
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X, y)

# =========================
# KPI CARDS (ZILLOW STYLE)
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Listings", len(df))
col2.metric("Avg Price", f"₹ {int(df['price'].mean()):,}")
col3.metric("Max Price", f"₹ {int(df['price'].max()):,}")
col4.metric("Min Price", f"₹ {int(df['price'].min()):,}")

st.markdown("---")

# =========================
# SIDEBAR INPUT (ZILLOW STYLE)
# =========================
st.sidebar.header("🏠 Property Details")

area = st.sidebar.slider("Area (sq ft)", 500, 5000, 1800)
bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3])
age = st.sidebar.slider("Property Age (years)", 0, 60, 10)
location_name = st.sidebar.selectbox("Location", ["city", "suburb", "rural"])
location = le.transform([location_name])[0]

# =========================
# PREDICTION
# =========================
input_data = np.array([[area, bedrooms, bathrooms, age, location]])
prediction = model.predict(input_data)[0]

st.sidebar.markdown("---")
st.sidebar.success(f"💰 Estimated Price: ₹ {int(prediction):,}")

# =========================
# TABS (PRO DASHBOARD)
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "📈 Analytics", "🤖 AI Prediction Insight"])

# =========================
# TAB 1 - MARKET OVERVIEW
# =========================
with tab1:
    st.subheader("Real Estate Market Overview")

    fig = px.histogram(df, x="price", nbins=50, title="House Price Distribution")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(df, x="area", y="price", color="location", title="Area vs Price")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 2 - ANALYTICS
# =========================
with tab2:
    st.subheader("Correlation Analysis")

    corr = df.corr(numeric_only=True)

    fig3 = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu"
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Feature Importance")

    importance = model.feature_importances_
    features = ["area", "bedrooms", "bathrooms", "age", "location"]

    fig4 = px.bar(
        x=importance,
        y=features,
        orientation="h",
        title="Impact on House Price"
    )
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# TAB 3 - PREDICTION INSIGHT
# =========================
with tab3:
    st.subheader("AI Prediction Engine")

    st.info("This model uses Random Forest Regression trained on synthetic real estate data.")

    st.metric("Predicted Price", f"₹ {int(prediction):,}")

    st.write("### Price Influence Breakdown")

    impact_df = pd.DataFrame({
        "Feature": ["Area", "Bedrooms", "Bathrooms", "Age", "Location"],
        "Importance": model.feature_importances_
    })

    fig5 = px.pie(impact_df, values="Importance", names="Feature", title="What affects price most?")
    st.plotly_chart(fig5, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Zillow-style AI Dashboard | Built with Streamlit + Machine Learning")

