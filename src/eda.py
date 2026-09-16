import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================
# DemandIQ - Exploratory Analysis
# ==============================

DATA_PATH = "data/processed/demand_daily.csv"
IMAGE_PATH = "images"

os.makedirs(IMAGE_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nStatistics:")
print(df.describe())

# ==============================
# 1. Sales Trend
# ==============================

plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["total_sales"]
)

plt.title("DemandIQ - Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    f"{IMAGE_PATH}/sales_trend.png",
    dpi=150
)

plt.show()

# ==============================
# 2. Monthly Sales
# ==============================

monthly_sales = (
    df.groupby("month")["total_sales"]
    .mean()
)

plt.figure(figsize=(10, 5))

monthly_sales.plot(kind="bar")

plt.title("Average Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.tight_layout()
plt.savefig(
    f"{IMAGE_PATH}/monthly_sales.png",
    dpi=150
)

plt.show()

# ==============================
# 3. Weekday vs Weekend
# ==============================

weekday_sales = (
    df.groupby("is_weekend")["total_sales"]
    .mean()
)

plt.figure(figsize=(8, 5))

weekday_sales.plot(kind="bar")

plt.title("Average Sales: Weekday vs Weekend")
plt.xlabel("Weekend (1 = Yes)")
plt.ylabel("Average Sales")

plt.tight_layout()
plt.savefig(
    f"{IMAGE_PATH}/weekday_weekend_sales.png",
    dpi=150
)

plt.show()

# ==============================
# 4. Promotion vs Sales
# ==============================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df.sample(min(3000, len(df))),
    x="total_promotion",
    y="total_sales"
)

plt.title("Promotions vs Sales")
plt.xlabel("Total Promotions")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig(
    f"{IMAGE_PATH}/promotion_vs_sales.png",
    dpi=150
)

plt.show()

# ==============================
# 5. Correlation
# ==============================

numeric_cols = [
    "total_sales",
    "total_promotion",
    "active_stores",
    "product_families",
    "oil_price",
    "is_holiday",
    "is_weekend"
]

corr = df[numeric_cols].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f"
)

plt.title("DemandIQ - Feature Correlation")

plt.tight_layout()
plt.savefig(
    f"{IMAGE_PATH}/correlation_heatmap.png",
    dpi=150
)

plt.show()

print("\n================================")
print("EDA COMPLETE")
print("================================")
print("Charts saved inside images/")