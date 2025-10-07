# ------------------------------------------------------
# EV Best Car Analysis (Top 5)
# Author: Alaa Miari 
# ------------------------------------------------------
# What it does:
# - Loads your EV CSV robustly (auto-detects separator , ; or tab)
# - Auto-detects key columns (model, price/cost, battery_kWh, age)
# - Aggregates per model and builds a Value Score
# - Exports ranked CSV and saves plots (Top 5 bar, scatter, correlation)
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # pyright: ignore[reportMissingModuleSource]
from pathlib import Path

# ======= USER CONFIG =======
CSV_PATH = Path("ev_charging_patterns.csv")  # <-- change if needed
OUT_DIR = Path("ev_results")                    # output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======= LOAD DATA (robust) =======
df = None
last_err = None
for sep in [",", ";", "\t"]:
    try:
        tmp = pd.read_csv(CSV_PATH, sep=sep, low_memory=False)
        if tmp.shape[1] >= 3:
            df = tmp
            print(f"✅ File loaded successfully with separator='{sep}'")
            break
    except Exception as e:
        last_err = e
        print(f"❌ Failed with separator='{sep}': {e}")

if df is None:
    raise RuntimeError(f"Could not read the CSV file at {CSV_PATH}. "
                       f"Last error: {last_err}")

# normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print("\nColumns in dataset:", df.columns.tolist())

# ======= COLUMN DETECTION =======
def find_col(candidates):
    for c in df.columns:
        for cand in candidates:
            if cand in c:
                return c
    return None

model_col   = find_col(["vehicle_model", "model", "car", "vehicle", "make_model", "name"])
price_col   = find_col(["price", "cost", "charging_cost", "msrp", "charging_cost_(usd)"])
battery_col = find_col(["battery_capacity_(kwh)", "battery_capacity", "kwh_capacity", "battery_kwh", "kwh"])
age_col     = find_col(["vehicle_age_(years)", "vehicle_age", "age_(years)", "age", "model_year", "year"])

print(f"\nDetected columns:")
print(f"  Model   : {model_col}")
print(f"  Price   : {price_col}")
print(f"  Battery : {battery_col}")
print(f"  Age     : {age_col}")

if model_col is None:
    raise ValueError("Could not find a model/vehicle column. "
                     "Please ensure a column name contains one of: 'model', 'vehicle', 'car', 'make_model'.")

if price_col is None and battery_col is None and age_col is None:
    raise ValueError("Need at least one metric (price/cost, battery capacity, or age) to rank cars.")

# ======= CLEAN NUMERIC =======
def to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    s = s.replace({"": np.nan, ".": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")

if price_col:   df[price_col]   = to_float(df[price_col])
if battery_col: df[battery_col] = to_float(df[battery_col])
if age_col:     df[age_col]     = to_float(df[age_col])

# ======= AGGREGATE PER MODEL =======
agg = {}
if price_col:   agg[price_col]   = "median"  # proxy for ongoing charging/session cost or purchase cost if present
if battery_col: agg[battery_col] = "mean"    # proxy for range potential
if age_col:     agg[age_col]     = "mean"    # newer is better

per_model = df.groupby(model_col, dropna=True).agg(agg).reset_index()

# if any of the numeric cols are entirely NaN after grouping, drop them from scoring
usable_cols = []
if price_col and per_model[price_col].notna().any(): usable_cols.append(price_col)
if battery_col and per_model[battery_col].notna().any(): usable_cols.append(battery_col)
if age_col and per_model[age_col].notna().any(): usable_cols.append(age_col)
if not usable_cols:
    raise ValueError("After cleaning, no usable numeric columns remained for scoring. Please inspect your CSV.")

# ======= SCORE CALCULATION =======
def normalize(series, invert=False):
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0 or x.max() == x.min():
        return pd.Series(np.nan, index=series.index)
    norm = (x - x.min()) / (x.max() - x.min())
    return 1 - norm if invert else norm

# Build normalized features
if price_col in per_model.columns:
    per_model["score_price"]   = normalize(per_model[price_col], invert=True)  # lower is better
else:
    per_model["score_price"]   = np.nan

if battery_col in per_model.columns:
    per_model["score_battery"] = normalize(per_model[battery_col])              # higher is better
else:
    per_model["score_battery"] = np.nan

if age_col in per_model.columns:
    per_model["score_age"]     = normalize(per_model[age_col], invert=True)     # newer is better
else:
    per_model["score_age"]     = np.nan

# weights (tune to your preference)
weights = {
    "score_price":   0.50,
    "score_battery": 0.30,
    "score_age":     0.20,
}
avail = [k for k in weights if k in per_model.columns and per_model[k].notna().any()]
wsum = sum(weights[k] for k in avail)
wnorm = {k: weights[k] / wsum for k in avail} if wsum > 0 else {}

per_model["value_score"] = 0.0
for k in avail:
    per_model[k] = per_model[k].fillna(per_model[k].mean())
    per_model["value_score"] += wnorm[k] * per_model[k]

# sort by value score
per_model = per_model.sort_values("value_score", ascending=False).reset_index(drop=True)

# ======= SAVE RESULTS =======
rank_csv = OUT_DIR / "ev_car_ranking.csv"
per_model.to_csv(rank_csv, index=False)
print(f"\n✅ Ranking saved to: {rank_csv}")

# ======= BASIC STATS =======
print("\n=== Ranked Summary (describe) ===")
print(per_model.describe(include='all').transpose())

# ======= PLOTS (Top 5) =======
plt.style.use("seaborn-v0_8-whitegrid")

# 1) Top 5 bar chart
topk = 5
top_df = per_model.head(topk)
plt.figure(figsize=(10, 6))
sns.barplot(x="value_score", y=model_col, data=top_df, palette="viridis")
plt.title(f"Top {topk} Electric Vehicles to Buy (Value Score)")
plt.xlabel("Value Score (0–1)")
plt.ylabel("Car Model")
plt.tight_layout()
plt.savefig(OUT_DIR / f"top{topk}_ev_bar.png")
plt.show()

# 2) Scatter: Battery vs Price (if both present)
if (price_col in per_model.columns) and (battery_col in per_model.columns):
    plt.figure(figsize=(9, 6))
    sc = sns.scatterplot(
        data=per_model,
        x=price_col,
        y=battery_col,
        hue="value_score",
        size="value_score",
        sizes=(20, 200),
        palette="cool",
        legend=False
    )
    plt.title("Battery Capacity vs Price (color/size = Value Score)")
    plt.xlabel("Price / Charging Cost (USD)")
    plt.ylabel("Battery Capacity (kWh)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "battery_vs_price.png")
    plt.show()

# 3) Correlation heatmap (only numeric columns that exist)
num_cols = [c for c in [price_col, battery_col, age_col, "value_score"] if c in per_model.columns]
num_df = per_model[num_cols].copy()
for c in num_cols:
    num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
if len(num_cols) >= 2:
    plt.figure(figsize=(6, 5))
    corr = num_df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Feature Correlations")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_heatmap.png")
    plt.show()

print("\n✅ All outputs saved in:", OUT_DIR)
print("   - ev_car_ranking.csv")
print(f"   - top{topk}_ev_bar.png")
if (price_col in per_model.columns) and (battery_col in per_model.columns):
    print("   - battery_vs_price.png")
print("   - correlation_heatmap.png (if enough numeric columns)")
