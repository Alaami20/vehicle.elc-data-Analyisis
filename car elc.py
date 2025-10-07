
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path("/Users/alaamiari/Downloads/ev_charging_patterns.csv")
OUT_PATH = Path("ev_car_recommendations.csv") # output ranking file

# -----------------------
# Helpers
# -----------------------
def coerce_numeric(series: pd.Series) -> pd.Series:
    """Strip currency/symbols and cast to float safely."""
    if series is None:
        return None
    s = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    s = s.replace({"": np.nan, ".": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")

def find_col(df: pd.DataFrame, candidates) -> str | None:
    """Return the first column containing any of the candidate substrings (case-insensitive)."""
    cols = [c.lower() for c in df.columns]
    for i, c in enumerate(cols):
        for cand in candidates:
            if cand in c:
                return df.columns[i]
    return None

def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalize to [0,1]. If invert=True, lower values become better (closer to 1)."""
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    xmin, xmax = x.min(), x.max()
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
        return pd.Series(np.nan, index=series.index)
    norm = (x - xmin) / (xmax - xmin)
    return 1 - norm if invert else norm

# -----------------------
# Load
# -----------------------
# Try a few encodings/separators just in case
df = None
for kwargs in [
    {"sep": ",", "encoding": "utf-8", "low_memory": False},
    {"sep": ";", "encoding": "utf-8", "low_memory": False},
    {"sep": ",", "encoding": "latin-1", "low_memory": False},
    {"sep": ";", "encoding": "latin-1", "low_memory": False},
]:
    try:
        tmp = pd.read_csv(CSV_PATH, **kwargs)
        if tmp.shape[1] >= 3 and tmp.shape[0] >= 5:
            df = tmp.copy()
            break
    except Exception:
        pass

if df is None:
    raise RuntimeError(f"Could not read {CSV_PATH}. Please check the file path/format.")

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# -----------------------
# Auto-detect columns
# -----------------------
model_col  = find_col(df, ["vehicle_model", "model", "car", "vehicle", "make_model", "name"])
cost_col   = find_col(df, ["charging_cost", "session_cost", "cost_usd", "price", "charging_cost_(usd)"])
batt_col   = find_col(df, ["battery_capacity_(kwh)", "battery_capacity", "kwh_capacity", "battery_kwh"])
age_col    = find_col(df, ["vehicle_age", "age_(years)", "vehicle_age_(years)", "model_year", "year"])

# Friendly check
print("Detected columns:")
print("  model :", model_col)
print("  cost  :", cost_col)
print("  batt  :", batt_col)
print("  age   :", age_col)

if model_col is None:
    raise ValueError("Could not find a model/vehicle column. Please rename your model column to include 'model' or 'vehicle' in its name.")

if cost_col is None and batt_col is None and age_col is None:
    raise ValueError("Could not find any of: charging cost, battery capacity, or vehicle age. "
                     "Please ensure at least one metric is present to rank cars.")

# -----------------------
# Clean numerics
# -----------------------
if cost_col is not None:
    df[cost_col] = coerce_numeric(df[cost_col])

if batt_col is not None:
    df[batt_col] = coerce_numeric(df[batt_col])

if age_col is not None:
    # Age might already be numeric; still coerce safely
    df[age_col] = coerce_numeric(df[age_col])

# -----------------------
# Aggregate per model
# -----------------------
agg_map = {}
if cost_col is not None:
    # Median cost per charging session as a proxy for ongoing cost
    agg_map[cost_col] = "median"
if batt_col is not None:
    # Mean battery capacity as a proxy for potential range
    agg_map[batt_col] = "mean"
if age_col is not None:
    # Mean age (years) — newer is better
    agg_map[age_col] = "mean"

per_model = df.groupby(model_col, dropna=True).agg(agg_map)
per_model = per_model.rename(columns={
    cost_col: "median_charging_cost_usd" if cost_col else None,
    batt_col: "mean_battery_kwh" if batt_col else None,
    age_col:  "mean_vehicle_age_years" if age_col else None
}).reset_index()

# -----------------------
# Build the Value Score
# -----------------------
# Normalize available features:
# - Lower cost is better  -> invert=True
# - Higher battery is better
# - Lower age is better   -> invert=True
scores = []
if "median_charging_cost_usd" in per_model.columns:
    per_model["n_cost"]  = minmax(per_model["median_charging_cost_usd"], invert=True)
else:
    per_model["n_cost"]  = np.nan

if "mean_battery_kwh" in per_model.columns:
    per_model["n_batt"]  = minmax(per_model["mean_battery_kwh"], invert=False)
else:
    per_model["n_batt"]  = np.nan

if "mean_vehicle_age_years" in per_model.columns:
    per_model["n_age"]   = minmax(per_model["mean_vehicle_age_years"], invert=True)
else:
    per_model["n_age"]   = np.nan

# Weights — tweak these to match your definition of “best”
weights = {
    "n_cost": 0.50,  # prioritize lower operating cost
    "n_batt": 0.30,  # favor bigger battery (range proxy)
    "n_age":  0.20,  # prefer newer cars
}

# Use only features that exist
avail = [k for k in weights if k in per_model.columns and per_model[k].notna().any()]
if not avail:
    raise ValueError("No usable features found to score. Check data content.")

wsum = sum(weights[k] for k in avail)
wnorm = {k: weights[k] / wsum for k in avail}

per_model["value_score"] = 0.0
for k in avail:
    # Fill missing with column mean to avoid dropping models
    per_model[k] = per_model[k].fillna(per_model[k].mean())
    per_model["value_score"] += wnorm[k] * per_model[k]

# -----------------------
# Sort & save
# -----------------------
# Arrange columns nicely
nice_cols = [model_col, "value_score"]
for c in ["median_charging_cost_usd", "mean_battery_kwh", "mean_vehicle_age_years", "n_cost", "n_batt", "n_age"]:
    if c in per_model.columns:
        nice_cols.append(c)

ranked = per_model[nice_cols].sort_values("value_score", ascending=False).reset_index(drop=True)

# Save output
ranked.to_csv(OUT_PATH, index=False)
print(f"\nSaved ranking to: {OUT_PATH.resolve()}\n")

# Print top 15 preview
print("Top picks:")
print(ranked.head(15).to_string(index=False))
