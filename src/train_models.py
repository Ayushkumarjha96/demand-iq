import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ==========================================
# DemandIQ - Model Training
# ==========================================

DATA_PATH = "data/processed/demand_features.csv"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)

print("Loading feature dataset...")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

# ==========================================
# Features and Target
# ==========================================

target = "total_sales"

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

X = df[features]
y = df[target]

# ==========================================
# Time-based Train/Test Split
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n==========================================")
print("TIME-BASED DATA SPLIT")
print("==========================================")

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows : {len(X_test):,}")

print(
    f"Training period: "
    f"{df['date'].iloc[0].date()} → "
    f"{df['date'].iloc[split_index - 1].date()}"
)

print(
    f"Testing period : "
    f"{df['date'].iloc[split_index].date()} → "
    f"{df['date'].iloc[-1].date()}"
)

# ==========================================
# Evaluation Function
# ==========================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) /
            np.maximum(np.abs(actual), 1)
        )
    ) * 100

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"MAPE : {mape:.2f}%")

    return {
        "model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2)
    }


results = []

# ==========================================
# 1. Naive Baseline
# ==========================================

print("\nTraining Naive Baseline...")

naive_predictions = df["sales_lag_1"].iloc[
    split_index:
].values

results.append(
    evaluate_model(
        "Naive Baseline",
        y_test,
        naive_predictions
    )
)

# ==========================================
# 2. Random Forest
# ==========================================

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions
    )
)

joblib.dump(
    rf,
    f"{MODEL_PATH}/random_forest_model.joblib"
)

# ==========================================
# 3. XGBoost
# ==========================================

print("\nTraining XGBoost...")

xgb = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

xgb_predictions = xgb.predict(X_test)

results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_predictions
    )
)

joblib.dump(
    xgb,
    f"{MODEL_PATH}/xgboost_model.joblib"
)

# ==========================================
# Compare Models
# ==========================================

results_df = pd.DataFrame(results)

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    results_df.to_string(index=False)
)

# ==========================================
# Select Best Model
# ==========================================

best_row = results_df.loc[
    results_df["MAE"].idxmin()
]

best_model_name = best_row["model"]

print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print(f"Selected model: {best_model_name}")

if best_model_name == "Random Forest":
    best_model = rf
elif best_model_name == "XGBoost":
    best_model = xgb
else:
    best_model = None

if best_model is not None:

    joblib.dump(
        best_model,
        f"{MODEL_PATH}/best_demand_model.joblib"
    )

# ==========================================
# Save Model Configuration
# ==========================================

config = {
    "target": target,
    "features": features,
    "best_model": best_model_name,
    "split_ratio": 0.80,
    "train_start": str(df["date"].iloc[0].date()),
    "train_end": str(
        df["date"].iloc[split_index - 1].date()
    ),
    "test_start": str(
        df["date"].iloc[split_index].date()
    ),
    "test_end": str(
        df["date"].iloc[-1].date()
    )
}

with open(
    f"{MODEL_PATH}/model_config.json",
    "w"
) as f:
    json.dump(config, f, indent=4)

results_df.to_csv(
    f"{MODEL_PATH}/model_comparison.csv",
    index=False
)

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print("Models saved in models/")
print("Model comparison saved.")
print("Configuration saved.")