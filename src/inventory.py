import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# DemandIQ - Inventory Intelligence
# ==========================================

DATA_PATH = "data/processed/demand_features.csv"
MODEL_PATH = "models/best_demand_model.joblib"
FORECAST_PATH = "reports/forecast_results.csv"

print("Loading DemandIQ data...")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

forecast = pd.read_csv(FORECAST_PATH)

# ==========================================
# Latest demand
# ==========================================

recent_demand = (
    df.sort_values("date")
    .tail(30)["total_sales"]
)

average_daily_demand = recent_demand.mean()
demand_std = recent_demand.std()

# ==========================================
# Forecast uncertainty
# ==========================================

forecast_errors = forecast["forecast_error"]

error_std = forecast_errors.std()

# Use the larger uncertainty estimate
demand_uncertainty = max(demand_std, error_std)

# ==========================================
# Latest ML forecast
# ==========================================

latest_forecast = forecast.iloc[-1]["predicted_sales"]

expected_daily_demand = (
    0.7 * latest_forecast
    + 0.3 * average_daily_demand
)

# ==========================================
# Inventory parameters
# ==========================================

lead_time_days = 7

service_level_z = 1.65
# Approximately 95% service level

# ==========================================
# Safety Stock
# ==========================================

safety_stock = (
    service_level_z
    * demand_uncertainty
    * np.sqrt(lead_time_days)
)

# ==========================================
# Reorder Point
# ==========================================

reorder_point = (
    expected_daily_demand * lead_time_days
    + safety_stock
)

# ==========================================
# Recommended Inventory
# ==========================================

recommended_inventory = (
    expected_daily_demand * 14
    + safety_stock
)

# ==========================================
# Stockout Risk
# ==========================================

if expected_daily_demand > average_daily_demand * 1.15:

    stockout_risk = "HIGH"

elif expected_daily_demand > average_daily_demand * 1.05:

    stockout_risk = "MEDIUM"

else:

    stockout_risk = "LOW"

# ==========================================
# Print Results
# ==========================================

print("\n==========================================")
print("DEMANDIQ INVENTORY INTELLIGENCE")
print("==========================================")

print(
    f"Average Daily Demand : "
    f"{average_daily_demand:,.0f}"
)

print(
    f"Latest ML Forecast   : "
    f"{latest_forecast:,.0f}"
)

print(
    f"Expected Daily Demand: "
    f"{expected_daily_demand:,.0f}"
)

print(
    f"Demand Uncertainty   : "
    f"{demand_uncertainty:,.0f}"
)

print(
    f"Lead Time            : "
    f"{lead_time_days} days"
)

print(
    f"Safety Stock         : "
    f"{safety_stock:,.0f}"
)

print(
    f"Reorder Point        : "
    f"{reorder_point:,.0f}"
)

print(
    f"Recommended Inventory: "
    f"{recommended_inventory:,.0f}"
)

print(
    f"Stockout Risk        : "
    f"{stockout_risk}"
)

# ==========================================
# Save Inventory Report
# ==========================================

inventory_result = pd.DataFrame({
    "metric": [
        "Average Daily Demand",
        "Latest ML Forecast",
        "Expected Daily Demand",
        "Demand Uncertainty",
        "Lead Time Days",
        "Safety Stock",
        "Reorder Point",
        "Recommended Inventory",
        "Stockout Risk"
    ],

    "value": [
        average_daily_demand,
        latest_forecast,
        expected_daily_demand,
        demand_uncertainty,
        lead_time_days,
        safety_stock,
        reorder_point,
        recommended_inventory,
        stockout_risk
    ]
})

os.makedirs("reports", exist_ok=True)

inventory_result.to_csv(
    "reports/inventory_recommendations.csv",
    index=False
)

print("\nSaved:")
print("reports/inventory_recommendations.csv")

print("\n==========================================")
print("INVENTORY ANALYSIS COMPLETE")
print("==========================================")