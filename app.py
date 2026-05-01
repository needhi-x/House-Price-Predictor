import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Zillow AI Dashboard",
    page_icon="🏡",
    layout="wide"
)

# =========================
# HEADER (ZILLOW STYLE)
# =========================
st.title("🏡 Zillow AI - Real Estate Price Predictor")
st.markdown("### Kaggle Dataset Powered ML Dashboard")
st.markdown("---")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    return df

df = load_data()

# =========================
# CLEAN DATA (KAGGLE)
# =========================
df = df[
    [
        "GrLivArea",
        "BedroomAbvGr",
        "FullBath",
        "YearBuilt",
        "OverallQual",
        "SalePrice"
    ]
].dropna()

features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "YearBuilt",
    "OverallQual"
]

X = df[features]
y = df["SalePrice"]

# =========================
# MODEL
# =========================
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X, y)

# =========================
# SIDEBAR (CLEAN ZILLOW STYLE)
# =========================
st.sidebar.header("🏠 Property Details")

area = st.sidebar.slider("Area (sq ft)", 500, 5000, 1500)
bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3])
age = st.sidebar.slider("Property Age (years)", 0, 150, 20)
quality = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)

# =========================
# CONVERT UI → MODEL FORMAT
# =========================
current_year = 2026
year_built = current_year - age

input_data = np.array([[area, bedrooms, bathrooms, year_built, quality]])
prediction = model.predict(input_data)[0]

st.sidebar.markdown("---")
st.sidebar.success(f"💰 Predicted Price: ₹ {int(prediction):,}")

# =========================
# KPI DASHBOARD
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Houses", len(df))
col2.metric("Avg Price", f"₹ {int(df['SalePrice'].mean()):,}")
col3.metric("Max Price", f"₹ {int(df['SalePrice'].max()):,}")
col4.metric("Min Price", f"₹ {int(df['SalePrice'].min()):,}")

st.markdown("---")

# =========================
# TABS (ZILLOW STYLE)
# =========================
tab1, tab2, tab3 = st.tabs([
    "📊 Market Overview",
    "📈 Analytics",
    "🤖 Prediction Insight"
])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("House Price Distribution")

    fig1 = px.histogram(df, x="SalePrice", nbins=50)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Area vs Price Relationship")

    fig2 = px.scatter(
        df,
        x="GrLivArea",
        y="SalePrice",
        color="OverallQual"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 2
# =========================
with tab2:
    st.subheader("Correlation Heatmap")

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

    fig4 = px.bar(
        x=importance,
        y=features,
        orientation="h"
    )

    st.plotly_chart(fig4, use_container_width=True)

# =========================
# TAB 3
# =========================
with tab3:
    st.subheader("AI Prediction Engine")

    st.metric("Estimated House Price", f"₹ {int(prediction):,}")

    avg_price = df["SalePrice"].mean()

    st.info("Model trained using Kaggle Housing Dataset (Random Forest)")

    # =========================
    # PIE CHART (NEW ADDITION)
    # =========================
    fig_pie = go.Figure(
        data=[go.Pie(
            labels=["Your Property Price", "Average Market Price"],
            values=[prediction, avg_price],
            hole=0.4
        )]
    )

    fig_pie.update_layout(title="Price Comparison vs Market Average")

    st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Zillow-style ML Dashboard | Kaggle Dataset | Streamlit")