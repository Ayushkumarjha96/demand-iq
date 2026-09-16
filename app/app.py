import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# DemandIQ - Sales Forecasting Dashboard
# ==========================================

st.set_page_config(
    page_title="DemandIQ",
    page_icon="📦",
    layout="wide"
)

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    color: #d1d5db;
    font-size: 17px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    text-align: center;
}

.metric-title {
    color: #6b7280;
    font-size: 14px;
}

.metric-value {
    font-size: 25px;
    font-weight: bold;
    margin-top: 5px;
}

.section-title {
    font-size: 25px;
    font-weight: bold;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Paths
# ==========================================

DATA_PATH = "data/processed/demand_features.csv"
FORECAST_PATH = "reports/forecast_results.csv"
MODEL_PATH = "models/best_demand_model.joblib"

# ==========================================
# Load Data
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    forecast = pd.read_csv(FORECAST_PATH)
    forecast["date"] = pd.to_datetime(forecast["date"])

    return df, forecast


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    df, forecast = load_data()
    model = load_model()

except Exception as e:

    st.error(f"Unable to load project files: {e}")
    st.stop()

# ==========================================
# Header
# ==========================================

st.markdown("""
<div class="hero">

<h1>📦 DemandIQ</h1>

<p>
AI-Powered Sales Forecasting & Inventory Intelligence
</p>

<p>
Forecast demand. Optimize inventory. Reduce stockout risk.
</p>

</div>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("⚙️ Inventory Settings")

lead_time = st.sidebar.slider(
    "Supplier Lead Time (days)",
    min_value=1,
    max_value=30,
    value=7
)

service_level = st.sidebar.slider(
    "Service Level",
    min_value=90,
    max_value=99,
    value=95
)

# Approximate Z-score
z_scores = {
    90: 1.28,
    91: 1.34,
    92: 1.41,
    93: 1.48,
    94: 1.55,
    95: 1.65,
    96: 1.75,
    97: 1.88,
    98: 2.05,
    99: 2.33
}

z = z_scores[service_level]

# ==========================================
# Core Calculations
# ==========================================

recent_demand = (
    df.sort_values("date")
    .tail(30)["total_sales"]
)

average_daily_demand = recent_demand.mean()

latest_forecast = forecast.iloc[-1]["predicted_sales"]

forecast_errors = forecast["forecast_error"]

demand_std = recent_demand.std()

error_std = forecast_errors.std()

demand_uncertainty = max(
    demand_std,
    error_std
)

expected_daily_demand = (
    0.7 * latest_forecast
    + 0.3 * average_daily_demand
)

safety_stock = (
    z
    * demand_uncertainty
    * np.sqrt(lead_time)
)

reorder_point = (
    expected_daily_demand * lead_time
    + safety_stock
)

recommended_inventory = (
    expected_daily_demand * 14
    + safety_stock
)

# ==========================================
# Stockout Risk
# ==========================================

demand_ratio = (
    expected_daily_demand /
    max(average_daily_demand, 1)
)

if demand_ratio >= 1.15:

    stockout_risk = "HIGH"

elif demand_ratio >= 1.05:

    stockout_risk = "MEDIUM"

else:

    stockout_risk = "LOW"

# ==========================================
# KPI Cards
# ==========================================

st.markdown(
    '<div class="section-title">📊 Business Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Daily Demand",
        f"{average_daily_demand:,.0f}"
    )

with col2:

    st.metric(
        "Latest ML Forecast",
        f"{latest_forecast:,.0f}"
    )

with col3:

    st.metric(
        "Recommended Inventory",
        f"{recommended_inventory:,.0f}"
    )

with col4:

    st.metric(
        "Stockout Risk",
        stockout_risk
    )

# ==========================================
# Forecast Chart
# ==========================================

st.markdown(
    '<div class="section-title">📈 Demand Forecast Performance</div>',
    unsafe_allow_html=True
)

chart_data = forecast.set_index("date")[
    ["actual_sales", "predicted_sales"]
]

st.line_chart(chart_data)

# ==========================================
# Forecast Metrics
# ==========================================

st.markdown(
    '<div class="section-title">🎯 Model Performance</div>',
    unsafe_allow_html=True
)

mae = np.mean(
    np.abs(
        forecast["actual_sales"]
        - forecast["predicted_sales"]
    )
)

rmse = np.sqrt(
    np.mean(
        (
            forecast["actual_sales"]
            - forecast["predicted_sales"]
        ) ** 2
    )
)

mape = np.mean(
    np.abs(
        (
            forecast["actual_sales"]
            - forecast["predicted_sales"]
        )
        /
        np.maximum(
            np.abs(forecast["actual_sales"]),
            1
        )
    )
) * 100

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "MAE",
        f"{mae:,.0f}"
    )

with c2:

    st.metric(
        "RMSE",
        f"{rmse:,.0f}"
    )

with c3:

    st.metric(
        "MAPE",
        f"{mape:.2f}%"
    )

# ==========================================
# Inventory Intelligence
# ==========================================

st.markdown(
    '<div class="section-title">📦 Inventory Intelligence</div>',
    unsafe_allow_html=True
)

inventory_col1, inventory_col2 = st.columns(2)

with inventory_col1:

    st.info(
        f"""
        **Expected Daily Demand**

        {expected_daily_demand:,.0f} units/day

        Based on recent demand and the latest
        machine-learning forecast.
        """
    )

    st.warning(
        f"""
        **Safety Stock**

        {safety_stock:,.0f} units

        Designed for approximately a
        {service_level}% service level.
        """
    )

with inventory_col2:

    st.success(
        f"""
        **Reorder Point**

        {reorder_point:,.0f} units

        Reorder when available inventory
        falls below this level.
        """
    )

    st.info(
        f"""
        **Recommended Inventory**

        {recommended_inventory:,.0f} units

        Covers approximately 14 days of
        expected demand plus safety stock.
        """
    )

# ==========================================
# Business Recommendation
# ==========================================

st.markdown(
    '<div class="section-title">💡 AI Business Recommendation</div>',
    unsafe_allow_html=True
)

if stockout_risk == "HIGH":

    st.error(
        """
        **High demand pressure detected.**

        Increase inventory coverage and consider
        accelerating replenishment to reduce
        potential stockout exposure.
        """
    )

elif stockout_risk == "MEDIUM":

    st.warning(
        """
        **Moderate demand pressure detected.**

        Monitor inventory closely and maintain
        sufficient safety stock during replenishment.
        """
    )

else:

    st.success(
        """
        **Demand appears relatively stable.**

        Current inventory planning parameters provide
        a reasonable buffer based on the selected
        lead time and service level.
        """
    )

# ==========================================
# Recent Forecast Data
# ==========================================

with st.expander("🔎 View Recent Forecast Data"):

    st.dataframe(
        forecast.tail(20),
        use_container_width=True
    )

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.caption(
    "DemandIQ | XGBoost Demand Forecasting • "
    "Time-Series Features • Inventory Optimization"
)