import pandas as pd
import os

# ==========================================
# DemandIQ - Feature Engineering
# ==========================================

INPUT_PATH = "data/processed/demand_daily.csv"
OUTPUT_PATH = "data/processed/demand_features.csv"

print("Loading processed data...")

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

# ==========================================
# Lag Features
# ==========================================

print("Creating lag features...")

df["sales_lag_1"] = df["total_sales"].shift(1)

df["sales_lag_7"] = df["total_sales"].shift(7)

df["sales_lag_14"] = df["total_sales"].shift(14)

df["sales_lag_28"] = df["total_sales"].shift(28)

# ==========================================
# Rolling Features
# IMPORTANT:
# Shift first to prevent data leakage
# ==========================================

print("Creating rolling features...")

df["sales_rolling_7"] = (
    df["total_sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["sales_rolling_14"] = (
    df["total_sales"]
    .shift(1)
    .rolling(14)
    .mean()
)

df["sales_rolling_28"] = (
    df["total_sales"]
    .shift(1)
    .rolling(28)
    .mean()
)

# ==========================================
# Additional Calendar Features
# ==========================================

df["quarter"] = df["date"].dt.quarter

df["day_of_year"] = df["date"].dt.dayofyear

df["week_of_month"] = (
    (df["date"].dt.day - 1) // 7 + 1
)

# ==========================================
# Promotion Features
# ==========================================

df["promotion_ratio"] = (
    df["total_promotion"] /
    (df["active_stores"] + 1)
)

# ==========================================
# Remove rows created by lagging
# ==========================================

df = df.dropna().reset_index(drop=True)

# ==========================================
# Save
# ==========================================

os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\n==========================================")
print("FEATURE ENGINEERING COMPLETE")
print("==========================================")

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nFeatures:")
print(df.columns.tolist())

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nSample:")
print(df.head())