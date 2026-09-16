import pandas as pd
import os

# ==============================
# DemandIQ - Data Preparation
# ==============================

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

print("Loading sales data...")

# Load main sales dataset
train = pd.read_csv(
    f"{RAW_PATH}/train.csv",
    usecols=["date", "store_nbr", "family", "sales", "onpromotion"]
)

print(f"Original rows: {len(train):,}")

# Convert date
train["date"] = pd.to_datetime(train["date"])

# ==============================
# Aggregate daily demand
# ==============================

print("Creating daily demand dataset...")

daily = (
    train.groupby("date")
    .agg(
        total_sales=("sales", "sum"),
        total_promotion=("onpromotion", "sum"),
        active_stores=("store_nbr", "nunique"),
        product_families=("family", "nunique")
    )
    .reset_index()
)

# ==============================
# Oil prices
# ==============================

print("Loading oil prices...")

oil = pd.read_csv(f"{RAW_PATH}/oil.csv")

oil["date"] = pd.to_datetime(oil["date"])

oil = oil.rename(columns={"dcoilwtico": "oil_price"})

oil = oil[["date", "oil_price"]]

# Fill missing oil prices
oil["oil_price"] = oil["oil_price"].interpolate()
oil["oil_price"] = oil["oil_price"].ffill().bfill()

# ==============================
# Holiday information
# ==============================

print("Loading holiday data...")

holidays = pd.read_csv(f"{RAW_PATH}/holidays_events.csv")

holidays["date"] = pd.to_datetime(holidays["date"])

# We only need whether a date has a holiday/event
holiday_dates = (
    holidays[["date"]]
    .drop_duplicates()
    .assign(is_holiday=1)
)

# ==============================
# Merge datasets
# ==============================

print("Merging datasets...")

daily = daily.merge(
    oil,
    on="date",
    how="left"
)

daily = daily.merge(
    holiday_dates,
    on="date",
    how="left"
)

daily["is_holiday"] = daily["is_holiday"].fillna(0).astype(int)

# ==============================
# Calendar features
# ==============================

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["day"] = daily["date"].dt.day
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)

daily["is_weekend"] = (
    daily["day_of_week"] >= 5
).astype(int)

# ==============================
# Sort by date
# ==============================

daily = daily.sort_values("date").reset_index(drop=True)

# ==============================
# Save processed data
# ==============================

output_path = f"{PROCESSED_PATH}/demand_daily.csv"

daily.to_csv(output_path, index=False)

print("\n================================")
print("DATA PREPARATION COMPLETE")
print("================================")
print(f"Rows: {len(daily):,}")
print(f"Columns: {len(daily.columns)}")
print(f"Saved to: {output_path}")

print("\nColumns:")
print(daily.columns.tolist())

print("\nFirst 5 rows:")
print(daily.head())