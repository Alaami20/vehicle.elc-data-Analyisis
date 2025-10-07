import pandas as pd

path = "ev_charging_patterns.csv"

for sep in [",", ";", "\t"]:
    try:
        df = pd.read_csv(path, sep=sep, nrows=5)
        print(f"\n✅ Read successful with separator='{sep}'")
        print(df.head())
    except Exception as e:
        print(f"\n❌ Failed with separator='{sep}': {e}")
