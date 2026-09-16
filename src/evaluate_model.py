import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ==========================================
# DemandIQ - Model Evaluation
# ==========================================

DATA_PATH = "data/processed/demand_features.csv"
MODEL_PATH = "models/best_demand_model.joblib"

os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Loading data...")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

features = [
    "total_promotion",
    "active_stores",
    "product_families",
    "oil_price",
    "is_holiday",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_7",
    "sales_rolling_14",
    "sales_rolling_28",
    "quarter",
    "day_of_year",
    "week_of_month",
    "promotion_ratio"
]

# ==========================================
# Test Data
# ==========================================

split_index = int(len(df) * 0.80)

X_test = df[features].iloc[split_index:]
y_test = df["total_sales"].iloc[split_index:]
dates = df["date"].iloc[split_index:]

print("Loading best model...")

model = joblib.load(MODEL_PATH)

print("Generating predictions...")

predictions = model.predict(X_test)

errors = y_test.values - predictions

# ==========================================
# Metrics
# ==========================================

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))

mape = np.mean(
    np.abs(errors) /
    np.maximum(np.abs(y_test.values), 1)
) * 100

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"MAPE : {mape:.2f}%")

# ==========================================
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    dates,
    y_test.values,
    label="Actual Sales"
)

plt.plot(
    dates,
    predictions,
    label="Predicted Sales"
)

plt.title("DemandIQ - Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/actual_vs_predicted.png",
    dpi=150
)

plt.close()

print("Saved actual_vs_predicted.png")

# ==========================================
# Forecast Error
# ==========================================

plt.figure(figsize=(14, 5))

plt.plot(
    dates,
    errors
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title("DemandIQ - Forecast Error")
plt.xlabel("Date")
plt.ylabel("Actual - Predicted")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/forecast_error.png",
    dpi=150
)

plt.close()

print("Saved forecast_error.png")

# ==========================================
# Feature Importance
# ==========================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

top_features = importance_df.head(15)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.title("DemandIQ - Top Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "images/feature_importance.png",
    dpi=150
)

plt.close()

print("Saved feature_importance.png")

# ==========================================
# Save Forecast Results
# ==========================================

forecast = pd.DataFrame({
    "date": dates.values,
    "actual_sales": y_test.values,
    "predicted_sales": predictions,
    "forecast_error": errors
})

forecast.to_csv(
    "reports/forecast_results.csv",
    index=False
)

# ==========================================
# Save Metrics
# ==========================================

metrics = pd.DataFrame({
    "metric": [
        "MAE",
        "RMSE",
        "MAPE"
    ],
    "value": [
        mae,
        rmse,
        mape
    ]
})

metrics.to_csv(
    "reports/evaluation_metrics.csv",
    index=False
)

print("\n==========================================")
print("EVALUATION COMPLETE")
print("==========================================")

print(f"Test rows: {len(y_test):,}")

print("\nFiles created:")
print("images/actual_vs_predicted.png")
print("images/forecast_error.png")
print("images/feature_importance.png")
print("reports/forecast_results.csv")
print("reports/evaluation_metrics.csv")