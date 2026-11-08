"""Check submission.csv before you submit: the same checks the challenge Space runs.

Usage: python code/check_submission.py [path/to/submission.csv]
Needs pandas, pyarrow and huggingface_hub (to read the list of cities from the training dataset):
pip install -r code/requirements.txt
"""

import sys

import pandas as pd

TRAIN = "hf://datasets/CMU-OSPO/protostar-climate-train-to-2018/city_info"
COLUMNS = ["city_id", "year", "metric", "predicted_value"]
YEARS = set(range(2019, 2026))
METRICS = {"annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"}

path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
cities = set(pd.read_parquet(TRAIN)["ID"])
df = pd.read_csv(path, dtype={"city_id": str, "metric": str})

if list(df.columns) != COLUMNS:
    sys.exit(f"Columns must be exactly: {','.join(COLUMNS)} (found: {','.join(df.columns)})")

year = pd.to_numeric(df["year"], errors="coerce")
value = pd.to_numeric(df["predicted_value"], errors="coerce")
checks = {
    "unknown city_id": ~df["city_id"].isin(cities),
    "year outside 2019-2025": ~year.isin(YEARS),
    "unknown metric": ~df["metric"].isin(METRICS),
    "missing or non-numeric predicted_value": value.isna() | value.abs().eq(float("inf")),
    "duplicate row": df.duplicated(["city_id", "year", "metric"]),
}
problems = [f"{mask.sum()} rows with {what}, e.g. line {mask.idxmax() + 2}" for what, mask in checks.items() if mask.any()]
expected = len(cities) * len(YEARS) * len(METRICS)
if not problems and len(df) != expected:
    problems.append(f"{len(df)} rows; expected {expected}")

if problems:
    sys.exit("Not ready:\n- " + "\n- ".join(problems))
print(f"OK: {path} has all {expected} forecasts and can be scored.")
